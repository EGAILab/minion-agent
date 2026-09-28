//! Minimal safe Wasmtime host for the exact `photon-node` 0.3.4 WASM exports used by Pi.
//! The binding follows the approved R005-A trace-replay harness, not a substitute image engine.

use std::sync::OnceLock;

use sha2::{Digest, Sha256};
use wasmtime::{
    Caller, Engine, ExternRef, ExternType, Func, Instance, Linker, Memory, Module, Ref, Store,
    Table, Val,
};

const WASM: &[u8] = include_bytes!("../../../assets/photon/photon_rs_bg.wasm");
const PINNED_HASH: &str = "10468181565c56004c867f3a4af96f89a0ef5a63a72f2b5fb12c1f1992a3615c";
static COMPILED: OnceLock<Result<(Engine, Module), String>> = OnceLock::new();

#[derive(Default)]
struct State {
    externref_table: Option<Table>,
}

fn failure(detail: impl AsRef<str>) -> String {
    format!("pinned Photon WASM failed: {}", detail.as_ref())
}

fn export_func(caller: &mut Caller<'_, State>, name: &str) -> wasmtime::Result<Func> {
    caller
        .get_export(name)
        .and_then(|export| export.into_func())
        .ok_or_else(|| wasmtime::Error::msg(format!("missing export {name}")))
}

fn memory(caller: &mut Caller<'_, State>) -> wasmtime::Result<Memory> {
    caller
        .get_export("memory")
        .and_then(|export| export.into_memory())
        .ok_or_else(|| wasmtime::Error::msg("missing memory"))
}

fn read_string(caller: &mut Caller<'_, State>, ptr: i32, len: i32) -> wasmtime::Result<String> {
    let mem = memory(caller)?;
    let mut buffer = vec![0u8; len as u32 as usize];
    mem.read(&mut *caller, ptr as u32 as usize, &mut buffer)?;
    Ok(String::from_utf8_lossy(&buffer).into_owned())
}

fn define_import(
    linker: &mut Linker<State>,
    import: &wasmtime::ImportType<'_>,
) -> wasmtime::Result<()> {
    let (module, name) = (import.module().to_owned(), import.name().to_owned());
    let ExternType::Func(ty) = import.ty() else {
        return Err(wasmtime::Error::msg(format!(
            "unexpected non-function import {name}"
        )));
    };
    let captured = name.clone();
    linker.func_new(
        &module,
        &name,
        ty,
        move |mut caller, args, results| match captured.as_str() {
            "__wbindgen_init_externref_table" => {
                let table = caller
                    .data()
                    .externref_table
                    .ok_or_else(|| wasmtime::Error::msg("missing externref table"))?;
                let offset = table.grow(&mut caller, 4, Ref::Extern(None))?;
                for (index, label) in [
                    (0, "undefined"),
                    (offset, "undefined"),
                    (offset + 1, "null"),
                    (offset + 2, "true"),
                    (offset + 3, "false"),
                ] {
                    let value = ExternRef::new(&mut caller, label)?;
                    table.set(&mut caller, index, Ref::Extern(Some(value)))?;
                }
                Ok(())
            }
            "__wbindgen_throw" => {
                let message = read_string(&mut caller, args[0].unwrap_i32(), args[1].unwrap_i32())?;
                Err(wasmtime::Error::msg(message))
            }
            "__wbg_new_abda76e883ba8a5f" => {
                results[0] = Val::ExternRef(Some(ExternRef::new(&mut caller, "Error")?));
                Ok(())
            }
            "__wbg_stack_658279fe44541cf6" => {
                let stack = b"Error\n    at <minion photon host>";
                let address = args[0].unwrap_i32() as u32 as usize;
                let malloc = export_func(&mut caller, "__wbindgen_malloc")?;
                let mut output = [Val::I32(0)];
                malloc.call(
                    &mut caller,
                    &[Val::I32(stack.len() as i32), Val::I32(1)],
                    &mut output,
                )?;
                let pointer = output[0].unwrap_i32() as u32;
                let mem = memory(&mut caller)?;
                mem.write(&mut caller, pointer as usize, stack)?;
                mem.write(&mut caller, address, &pointer.to_le_bytes())?;
                mem.write(
                    &mut caller,
                    address + 4,
                    &(stack.len() as u32).to_le_bytes(),
                )?;
                Ok(())
            }
            "__wbg_error_f851667af71bcfc6" => {
                let (ptr, len) = (args[0].unwrap_i32(), args[1].unwrap_i32());
                let _message = read_string(&mut caller, ptr, len)?;
                let free = export_func(&mut caller, "__wbindgen_free")?;
                free.call(
                    &mut caller,
                    &[Val::I32(ptr), Val::I32(len), Val::I32(1)],
                    &mut [],
                )?;
                Ok(())
            }
            "__wbindgen_memory" => {
                results[0] = Val::ExternRef(Some(ExternRef::new(&mut caller, "memory")?));
                Ok(())
            }
            other => Err(wasmtime::Error::msg(format!(
                "unexpected Photon import {other}"
            ))),
        },
    )?;
    Ok(())
}

pub(super) struct Photon {
    store: Store<State>,
    instance: Instance,
}

impl Photon {
    pub fn new() -> Result<Self, String> {
        let (engine, module) = COMPILED
            .get_or_init(|| {
                let hash = Sha256::digest(WASM);
                let hash: String = hash.iter().map(|byte| format!("{byte:02x}")).collect();
                if hash != PINNED_HASH {
                    return Err(failure("artifact hash differs from photon-node 0.3.4"));
                }
                let engine = Engine::default();
                let module =
                    Module::new(&engine, WASM).map_err(|error| failure(error.to_string()))?;
                Ok((engine, module))
            })
            .as_ref()
            .map_err(Clone::clone)?;
        let mut linker = Linker::new(engine);
        for import in module.imports() {
            define_import(&mut linker, &import).map_err(|error| failure(error.to_string()))?;
        }
        let mut store = Store::new(engine, State::default());
        let instance = linker
            .instantiate(&mut store, module)
            .map_err(|error| failure(error.to_string()))?;
        store.data_mut().externref_table = instance.get_table(&mut store, "__wbindgen_export_2");
        let mut host = Self { store, instance };
        host.call("__wbindgen_start", &[])?;
        Ok(host)
    }

    fn call(&mut self, name: &str, args: &[Val]) -> Result<Vec<Val>, String> {
        let function = self
            .instance
            .get_func(&mut self.store, name)
            .ok_or_else(|| failure(format!("missing export {name}")))?;
        let mut results = vec![Val::I32(0); function.ty(&self.store).results().len()];
        function
            .call(&mut self.store, args, &mut results)
            .map_err(|error| failure(error.to_string()))?;
        Ok(results)
    }

    fn pass_bytes(&mut self, bytes: &[u8]) -> Result<(i32, i32), String> {
        let ptr = self.call(
            "__wbindgen_malloc",
            &[Val::I32(bytes.len() as i32), Val::I32(1)],
        )?[0]
            .unwrap_i32();
        let memory = self
            .instance
            .get_memory(&mut self.store, "memory")
            .ok_or_else(|| failure("missing memory"))?;
        memory
            .write(&mut self.store, ptr as u32 as usize, bytes)
            .map_err(|error| failure(error.to_string()))?;
        Ok((ptr, bytes.len() as i32))
    }

    fn take_bytes(&mut self, result: &[Val]) -> Result<Vec<u8>, String> {
        let (ptr, len) = (result[0].unwrap_i32(), result[1].unwrap_i32());
        let memory = self
            .instance
            .get_memory(&mut self.store, "memory")
            .ok_or_else(|| failure("missing memory"))?;
        let mut bytes = vec![0u8; len as u32 as usize];
        memory
            .read(&self.store, ptr as u32 as usize, &mut bytes)
            .map_err(|error| failure(error.to_string()))?;
        self.call(
            "__wbindgen_free",
            &[Val::I32(ptr), Val::I32(len), Val::I32(1)],
        )?;
        Ok(bytes)
    }

    pub fn decode(&mut self, bytes: &[u8]) -> Result<i32, String> {
        let (ptr, len) = self.pass_bytes(bytes)?;
        Ok(self.call(
            "photonimage_new_from_byteslice",
            &[Val::I32(ptr), Val::I32(len)],
        )?[0]
            .unwrap_i32())
    }

    pub fn new_image(&mut self, pixels: &[u8], width: i32, height: i32) -> Result<i32, String> {
        let (ptr, len) = self.pass_bytes(pixels)?;
        Ok(self.call(
            "photonimage_new",
            &[
                Val::I32(ptr),
                Val::I32(len),
                Val::I32(width),
                Val::I32(height),
            ],
        )?[0]
            .unwrap_i32())
    }

    pub fn width(&mut self, image: i32) -> Result<i32, String> {
        Ok(self.call("photonimage_get_width", &[Val::I32(image)])?[0].unwrap_i32())
    }
    pub fn height(&mut self, image: i32) -> Result<i32, String> {
        Ok(self.call("photonimage_get_height", &[Val::I32(image)])?[0].unwrap_i32())
    }
    pub fn raw_pixels(&mut self, image: i32) -> Result<Vec<u8>, String> {
        let output = self.call("photonimage_get_raw_pixels", &[Val::I32(image)])?;
        self.take_bytes(&output)
    }
    pub fn png(&mut self, image: i32) -> Result<Vec<u8>, String> {
        let output = self.call("photonimage_get_bytes", &[Val::I32(image)])?;
        self.take_bytes(&output)
    }
    pub fn jpeg(&mut self, image: i32, quality: i32) -> Result<Vec<u8>, String> {
        let output = self.call(
            "photonimage_get_bytes_jpeg",
            &[Val::I32(image), Val::I32(quality)],
        )?;
        self.take_bytes(&output)
    }
    pub fn resize(&mut self, image: i32, width: i32, height: i32) -> Result<i32, String> {
        Ok(self.call(
            "resize",
            &[
                Val::I32(image),
                Val::I32(width),
                Val::I32(height),
                Val::I32(5),
            ],
        )?[0]
            .unwrap_i32())
    }
    pub fn flip_horizontal(&mut self, image: i32) -> Result<(), String> {
        self.call("fliph", &[Val::I32(image)])?;
        Ok(())
    }
    pub fn flip_vertical(&mut self, image: i32) -> Result<(), String> {
        self.call("flipv", &[Val::I32(image)])?;
        Ok(())
    }
    pub fn free(&mut self, image: i32) {
        let _ = self.call("__wbg_photonimage_free", &[Val::I32(image), Val::I32(0)]);
    }
}
