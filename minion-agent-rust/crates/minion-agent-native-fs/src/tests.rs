use super::valid_directory_link;

fn link(tag: u32, name: &str) -> Vec<u8> {
    let mut data = vec![0, 0];
    let units = name.encode_utf16().collect::<Vec<_>>();
    data.extend_from_slice(&((units.len() * 2) as u16).to_le_bytes());
    data.extend_from_slice(&[0; 4]);
    if tag == 0xa000000c {
        data.extend_from_slice(&[0; 4]);
    }
    for u in units {
        data.extend_from_slice(&u.to_le_bytes());
    }
    let mut bytes = tag.to_le_bytes().to_vec();
    bytes.extend_from_slice(&(data.len() as u16).to_le_bytes());
    bytes.extend_from_slice(&[0; 2]);
    bytes.extend(data);
    bytes
}

#[test]
fn directory_reparse_payloads_keep_libuv_readlink_acceptance() {
    assert!(valid_directory_link(&link(0xa000000c, "relative")));
    assert!(valid_directory_link(&link(0xa0000003, r"\??\E:\inside")));
    assert!(!valid_directory_link(&link(
        0xa0000003,
        r"\??\Volume{guid}"
    )));
    assert!(!valid_directory_link(&link(0x9000001a, "cloud")));
    assert!(!valid_directory_link(&[]));
    let mut malformed = link(0xa000000c, "x");
    malformed[8] = 255;
    assert!(!valid_directory_link(&malformed));
    let strings = "package\0entry\0E:\\target\0";
    let mut data = 3u32.to_le_bytes().to_vec();
    for u in strings.encode_utf16() {
        data.extend_from_slice(&u.to_le_bytes());
    }
    let mut app = 0x8000001bu32.to_le_bytes().to_vec();
    app.extend_from_slice(&(data.len() as u16).to_le_bytes());
    app.extend_from_slice(&[0; 2]);
    app.extend(data);
    assert!(valid_directory_link(&app));
    app[8] = 2;
    assert!(!valid_directory_link(&app));
}
