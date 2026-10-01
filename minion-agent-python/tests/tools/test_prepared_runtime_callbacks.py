"""CE-L0506-D001-I001-01 callback dimensions (`TOOL-041`) for a pydantic-model tool:
- `CE-I001-C001`: the finite rule replays no user validator. Each runs exactly once, in pydantic's
  order, with its own result.
- `CE-I001-C002`: the rule judges the value Layer 06 delivers. A non-finite value a callback
  produces in a declared numeric position is rejected before the before-hook or `execute()` sees it.

Each case runs through the real pipeline (`execute_call`)."""

from __future__ import annotations

import enum
import math
from datetime import datetime
from typing import Annotated, Any, NewType

import pytest
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic import ValidationError as PydanticValidationError

from minion_agent.llm import TextBlock, ToolCallBlock
from minion_agent.runtime import Context
from minion_agent.tools import execute as execute_module
from minion_agent.tools.definition import ToolDefinition
from minion_agent.tools.events import TOOLS_PRE_EXECUTE, declare_tools_events
from minion_agent.tools.execute import execute_call
from minion_agent.tools.registry import ToolRegistry
from minion_agent.tools.result import ToolResult

NON_FINITE = [math.inf, -math.inf, math.nan]


async def _run(model: type[BaseModel], arguments: dict[str, Any]) -> dict[str, Any]:
    """One real call: `{rejected, text, hook, execute}`. `hook`/`execute` are the arguments each
    saw, or None when it never ran."""
    seen: dict[str, Any] = {"hook": None, "execute": None}

    async def execute(tool_call_id: str, delivered: dict[str, Any]) -> ToolResult:
        seen["execute"] = delivered
        return ToolResult(tool_call_id=tool_call_id, content=(TextBlock(text="ok"),), tool_name="p")

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(name="p", label="p", description="d", parameters=model, execute=execute)
    )
    ctx = Context()
    declare_tools_events(ctx.events)

    async def hook(call: Any, d: Any, delivered: Any, signal: Any, next_: Any) -> Any:
        seen["hook"] = delivered
        return await next_()

    ctx.events.on(TOOLS_PRE_EXECUTE, hook)
    result = await execute_call(
        ToolCallBlock(id="c", name="p", arguments=arguments), registry=registry, ctx=ctx
    )
    first = result.content[0]
    text = first.text if isinstance(first, TextBlock) else ""
    return {"rejected": result.is_error and "invalid arguments" in text, "text": text, **seen}


def _assert_rejected(outcome: dict[str, Any]) -> None:
    assert outcome["rejected"], outcome
    assert "finite number" in outcome["text"]
    assert outcome["hook"] is None and outcome["execute"] is None


# --- CE-I001-C001: each user validator runs exactly once --------------------------------------

TRACE: list[str] = []


class OneShotInner(BaseModel):
    y: float

    @field_validator("y")
    @classmethod
    def once(cls, value: float) -> float:
        TRACE.append(f"inner.y={value}")
        if TRACE.count(f"inner.y={value}") > 1:
            raise ValueError("already consumed")
        return value * 2


class OneShot(BaseModel):
    x: float
    inner: OneShotInner
    either: list[float] | list[Any] = []

    @field_validator("x", mode="before")
    @classmethod
    def before(cls, value: Any) -> Any:
        TRACE.append(f"before x={value}")
        return value

    @field_validator("x")
    @classmethod
    def after(cls, value: float) -> float:
        TRACE.append(f"after x={value}")
        if TRACE.count(f"after x={value}") > 1:
            raise ValueError("already consumed")
        return value + 1

    @model_validator(mode="after")
    def whole(self) -> OneShot:
        TRACE.append("model")
        return self


@pytest.fixture(autouse=True)
def _fresh_trace() -> None:
    TRACE.clear()


async def test_each_user_validator_runs_once_in_order_with_its_own_result() -> None:
    outcome = await _run(OneShot, {"x": 1.0, "inner": {"y": -0.0}})
    assert not outcome["rejected"], outcome
    assert TRACE == ["before x=1.0", "after x=1.0", "inner.y=-0.0", "model"]
    assert outcome["execute"] == {"x": 2.0, "inner": {"y": -0.0}, "either": []}
    assert math.copysign(1.0, outcome["execute"]["inner"]["y"]) == -1.0
    assert outcome["hook"] == outcome["execute"]


async def test_a_rejected_non_finite_still_ran_each_validator_once() -> None:
    _assert_rejected(await _run(OneShot, {"x": math.inf, "inner": {"y": 1.0}}))
    assert TRACE == ["before x=inf", "after x=inf", "inner.y=1.0", "model"]


async def test_an_unconstrained_alternative_keeps_non_finite_with_one_validation() -> None:
    outcome = await _run(OneShot, {"x": 1.0, "inner": {"y": 1.0}, "either": [math.inf]})
    assert not outcome["rejected"], outcome
    assert outcome["execute"]["either"] == [math.inf]
    assert TRACE == ["before x=1.0", "after x=1.0", "inner.y=1.0", "model"]


async def test_input_finite_variant_replay_is_killed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative control: the rejected checkpoint proposal. A second, input-side validation through
    a variant that inherits the user validators replays them."""

    def replay(model: type[BaseModel], delivered: dict[str, Any]) -> None:
        del delivered
        try:
            model.model_validate(ARGUMENTS)
        except PydanticValidationError as error:
            raise execute_module.ArgumentValidationError(str(error)) from error

    ARGUMENTS = {"x": 1.0, "inner": {"y": 1.0}}
    monkeypatch.setattr(execute_module, "_reject_declared_non_finite", replay)
    outcome = await _run(OneShot, ARGUMENTS)
    assert outcome["rejected"] and "already consumed" in outcome["text"]


# --- CE-I001-C002: a callback-produced non-finite value in a declared numeric position ---------

VALUE: list[float] = [math.inf]


class FieldAfter(BaseModel):
    field: float

    @field_validator("field")
    @classmethod
    def rewrite(cls, value: float) -> float:
        return VALUE[0]


class FieldBefore(BaseModel):
    field: float

    @field_validator("field", mode="before")
    @classmethod
    def parse(cls, value: Any) -> Any:
        return VALUE[0] if value == "big" else value


class ModelAfter(BaseModel):
    field: float = 0.0

    @model_validator(mode="after")
    def rewrite(self) -> ModelAfter:
        self.field = VALUE[0]
        return self


class AnnotatedAfter(BaseModel):
    field: Annotated[float, AfterValidator(lambda value: VALUE[0])]


class NestedAfter(BaseModel):
    items: list[FieldAfter]
    maybe: FieldAfter | None = None


@pytest.fixture(params=NON_FINITE, ids=["+inf", "-inf", "nan"])
def produced(request: pytest.FixtureRequest) -> Any:
    VALUE[0] = request.param
    yield request.param
    VALUE[0] = math.inf


@pytest.mark.parametrize(
    ("model", "arguments"),
    [
        (FieldAfter, {"field": 1.0}),
        (FieldBefore, {"field": "big"}),
        (ModelAfter, {}),
        (AnnotatedAfter, {"field": 1.0}),
        (NestedAfter, {"items": [{"field": 1.0}]}),
        (NestedAfter, {"items": [], "maybe": {"field": 1.0}}),
    ],
    ids=["field-after", "field-before", "model-after", "annotated-after", "nested", "nullable"],
)
async def test_a_callback_produced_non_finite_declared_number_is_rejected(
    model: type[BaseModel], arguments: dict[str, Any], produced: float
) -> None:
    del produced
    _assert_rejected(await _run(model, arguments))


class OpenAfter(BaseModel):
    loose: float | Any = 0.0
    text: str = ""

    @model_validator(mode="after")
    def rewrite(self) -> OpenAfter:
        self.loose = VALUE[0]
        self.text = VALUE[0]  # type: ignore[assignment]  # outside the declared shape
        return self


@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings")  # str field holds a float
async def test_a_callback_produced_non_finite_outside_a_declared_number_is_kept(
    produced: float,
) -> None:
    outcome = await _run(OpenAfter, {})
    assert not outcome["rejected"], outcome
    assert repr(outcome["execute"]) == repr({"loose": produced, "text": produced})


class Clamped(BaseModel):
    field: float

    @field_validator("field")
    @classmethod
    def clamp(cls, value: float) -> float:
        return value if math.isfinite(value) else 0.0


async def test_a_callback_that_makes_the_delivered_value_finite_is_accepted() -> None:
    """The rule constrains the delivered value; a user callback is pydantic's authority."""
    outcome = await _run(Clamped, {"field": math.inf})
    assert not outcome["rejected"], outcome
    assert outcome["execute"] == {"field": 0.0}


# --- declared types outside the matrix -------------------------------------------------------

Meters = NewType("Meters", float)
type Distance = float | None


class Colour(enum.Enum):
    RED = 1.5


class Generic_[T](BaseModel):
    value: T


class Root(RootModel[list[float]]):
    pass


class Computed(BaseModel):
    base: float = 1.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def scaled(self) -> float:
        return self.base * VALUE[0]


class ByAlias(BaseModel):
    model_config = ConfigDict(serialize_by_alias=True, populate_by_name=True)
    limit: float = Field(0.0, alias="max-limit")
    other: float = Field(0.0, serialization_alias="other-limit")

    @computed_field(alias="doubled")  # type: ignore[prop-decorator]
    @property
    def twice(self) -> float:
        return self.limit * 2


class Kinds(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    alias_: Distance = Field(None, alias="alias")
    meters: Meters = Meters(0.0)
    colour: Colour = Colour.RED
    when: datetime = datetime(2026, 1, 1)
    unbound: Generic_ = Field(default_factory=lambda: Generic_(value=0.0))  # type: ignore[type-arg]
    root: Root = Field(default_factory=lambda: Root([]))


@pytest.mark.parametrize(
    ("model", "arguments", "rejected"),
    [
        (Kinds, {"alias": math.inf}, True),
        (Kinds, {"meters": math.inf}, True),
        (Kinds, {"root": [1.0, math.inf]}, True),
        (Kinds, {"unbound": {"value": math.inf}}, False),
        (Kinds, {"colour": Colour.RED, "when": datetime(2026, 1, 2), "alias": -0.0}, False),
        (Computed, {"base": 1.0}, True),
        (Computed, {"base": 0.5}, True),
        (ByAlias, {"max-limit": math.inf}, True),
        (ByAlias, {"other": math.inf}, True),
        (ByAlias, {"limit": 2.0}, False),
    ],
    ids=[
        "type-alias",
        "new-type",
        "root-model",
        "type-var",
        "enum-datetime",
        "computed",
        "computed-half",
        "serialize-by-alias",
        "serialization-alias",
        "by-alias-finite",
    ],
)
async def test_declared_type_kinds(
    model: type[BaseModel], arguments: dict[str, Any], rejected: bool
) -> None:
    outcome = await _run(model, arguments)
    if rejected:
        _assert_rejected(outcome)
    else:
        assert not outcome["rejected"], outcome


# --- CE-I001-C002 refined: an out-of-shape position never exempts a numeric one ----------------


class NumericOnly(BaseModel):
    number: float = 0.0

    @model_validator(mode="after")
    def rewrite(self) -> NumericOnly:
        self.number = VALUE[0]
        return self


class WithMalformedSibling(BaseModel):
    number: float = 0.0
    text: str = "ok"

    @model_validator(mode="after")
    def rewrite(self) -> WithMalformedSibling:
        self.number = VALUE[0]
        self.text = VALUE[0]  # type: ignore[assignment]  # outside this non-numeric field's shape
        return self


class Count(BaseModel):
    count: int = 0

    @model_validator(mode="after")
    def rewrite(self) -> Count:
        self.count = VALUE[0]  # type: ignore[assignment]  # a non-finite in a declared integer
        return self


class Pair(BaseModel):
    number: float = 0.0
    text: str = "ok"


class NestedMalformed(BaseModel):
    pair: Pair = Field(default_factory=Pair)
    pairs: list[Pair] = []

    @model_validator(mode="after")
    def rewrite(self) -> NestedMalformed:
        self.pair.number = VALUE[0]
        self.pair.text = VALUE[0]  # type: ignore[assignment]
        return self


class ListMalformed(BaseModel):
    numbers: list[float] = []
    maybe: Pair | None = None

    @model_validator(mode="after")
    def rewrite(self) -> ListMalformed:
        self.numbers = [VALUE[0], "x"]  # type: ignore[list-item]  # a numeric element + a mismatch
        return self


class OpenWithMalformedSibling(BaseModel):
    loose: float | Any = 0.0
    items: list[float] | list[Any] = []
    text: str = "ok"

    @model_validator(mode="after")
    def rewrite(self) -> OpenWithMalformedSibling:
        self.loose = VALUE[0]
        self.items = [VALUE[0]]
        self.text = VALUE[0]  # type: ignore[assignment]
        return self


@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings")
@pytest.mark.parametrize(
    "model",
    [NumericOnly, WithMalformedSibling, Count, NestedMalformed, ListMalformed],
    ids=["numeric-only", "malformed-sibling", "integer", "nested-sibling", "list-element"],
)
async def test_an_out_of_shape_position_never_exempts_a_numeric_one(
    model: type[BaseModel], produced: float
) -> None:
    del produced
    _assert_rejected(await _run(model, {}))


@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings")
async def test_unconstrained_alternatives_beside_a_malformed_sibling_are_kept(
    produced: float,
) -> None:
    """No blanket sweep: only a declared numeric position rejects."""
    outcome = await _run(OpenWithMalformedSibling, {})
    assert not outcome["rejected"], outcome
    assert repr(outcome["execute"]) == repr(
        {"loose": produced, "items": [produced], "text": produced}
    )


@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings")
async def test_whole_record_open_shape_exemption_is_killed(
    monkeypatch: pytest.MonkeyPatch, produced: float
) -> None:
    """Negative control: rev 2's rule -- reject only when the WHOLE delivered value fits the
    non-finite-admitting shape -- lets a malformed sibling exempt the numeric field."""
    finite_rule = execute_module._reject_declared_non_finite

    def whole_record_exemption(model: type[BaseModel], delivered: dict[str, Any]) -> None:
        open_values = _map_non_finite(delivered)
        try:
            execute_module._record_shape(model).model_validate(open_values)
        except PydanticValidationError:
            return
        finite_rule(model, delivered)

    monkeypatch.setattr(execute_module, "_reject_declared_non_finite", whole_record_exemption)
    assert not (await _run(WithMalformedSibling, {}))["rejected"]
    del produced


def _map_non_finite(value: Any) -> Any:
    """The delivered value with every non-finite number made finite: validating it against the
    finite shape is validating the original against the non-finite-admitting shape."""
    if isinstance(value, dict):
        return {key: _map_non_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_map_non_finite(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return 0.0
    return value


# --- L0506-D001-I002: only an actual runtime number is judged ---------------------------------

TEXT: list[str] = ["Infinity"]


class NumericLookingString(BaseModel):
    number: float

    @field_validator("number")
    @classmethod
    def rewrite(cls, value: float) -> Any:
        return TEXT[0]


class StringBesideNumber(BaseModel):
    number: float = 0.0
    count: int = 0
    other: float = 0.0
    items: list[float] = []
    either: float | str = 0.0

    @model_validator(mode="after")
    def rewrite(self) -> StringBesideNumber:
        self.number = TEXT[0]  # type: ignore[assignment]
        self.count = TEXT[0]  # type: ignore[assignment]
        self.items = [TEXT[0]]  # type: ignore[list-item]
        self.either = TEXT[0]
        self.other = VALUE[0]
        return self


class StringsOnly(BaseModel):
    number: float = 0.0
    count: int = 0
    items: list[float] = []
    inner: Pair = Field(default_factory=Pair)

    @model_validator(mode="after")
    def rewrite(self) -> StringsOnly:
        self.number = TEXT[0]  # type: ignore[assignment]
        self.count = TEXT[0]  # type: ignore[assignment]
        self.items = [TEXT[0]]  # type: ignore[list-item]
        self.inner.number = TEXT[0]  # type: ignore[assignment]
        return self


@pytest.fixture(params=["Infinity", "-Infinity", "NaN", "outside", "1"])
def text(request: pytest.FixtureRequest) -> Any:
    TEXT[0] = request.param
    yield request.param
    TEXT[0] = "Infinity"


@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings")
async def test_a_delivered_string_in_a_numeric_field_is_not_a_number(text: str) -> None:
    """The delivered value is a string, not a runtime number: the check never coerces it."""
    outcome = await _run(NumericLookingString, {"number": 1.0})
    assert not outcome["rejected"], outcome
    assert outcome["hook"] == outcome["execute"] == {"number": text}


@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings")
async def test_strings_in_numeric_positions_are_delivered_unchanged(text: str) -> None:
    outcome = await _run(StringsOnly, {})
    assert not outcome["rejected"], outcome
    assert outcome["execute"] == {
        "number": text,
        "count": text,
        "items": [text],
        "inner": {"number": text, "text": "ok"},
    }


@pytest.mark.filterwarnings("ignore:Pydantic serializer warnings")
async def test_a_string_never_exempts_a_genuine_non_finite_sibling(
    text: str, produced: float
) -> None:
    del text, produced
    _assert_rejected(await _run(StringBesideNumber, {}))
