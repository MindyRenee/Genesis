use std::fs::File;
use std::os::unix::io::RawFd;
use nix::fcntl::OFlag;
use nix::sys::stat::Mode;
use memmap2::MmapMut;
use crate::error::ShmemError;

pub type ShmemConfExt = ();
pub type SharedMemError = ShmemError;

pub struct UnixMapData {
    pub file: File,
    pub mmap_data: MmapMut,
    pub map_size: usize,
    pub unique_id: String,
    is_owner: bool,
}

pub type MapData = UnixMapData;

#[cfg(target_os = "android")]
pub fn shm_open(name: &str, flag: OFlag, mode: Mode) -> nix::Result<RawFd> {
    let mut path = std::env::temp_dir();
    let clean_name = name.trim_start_matches('/');
    path.push(clean_name);
    nix::fcntl::open(&path, flag, mode)
}

#[cfg(target_os = "android")]
pub fn shm_unlink(name: &str) -> nix::Result<()> {
    let mut path = std::env::temp_dir();
    let clean_name = name.trim_start_matches('/');
    path.push(clean_name);
    std::fs::remove_file(path).map_err(|_| nix::Error::from_errno(nix::errno::Errno::ENOENT))
}

#[cfg(not(target_os = "android"))]
use nix::sys::mman::{shm_open, shm_unlink};

pub fn create_mapping(id: &str, size: usize, _ext: &ShmemConfExt) -> Result<UnixMapData, ShmemError> {
    let mut path = std::env::temp_dir();
    path.push(id.trim_start_matches('/'));
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(&path)
        .map_err(|e| ShmemError::MapCreateFailed(e.to_string()))?;
    file.set_len(size as u64)
        .map_err(|e| ShmemError::MapCreateFailed(e.to_string()))?;
    let mmap_data = unsafe {
        MmapMut::map_mut(&file).map_err(|e| ShmemError::MapCreateFailed(e.to_string()))?
    };
    Ok(UnixMapData {
        file,
        mmap_data,
        map_size: size,
        unique_id: id.to_string(),
        is_owner: true,
    })
}

pub fn open_mapping(id: &str, size: usize, _ext: &ShmemConfExt) -> Result<UnixMapData, ShmemError> {
    let mut path = std::env::temp_dir();
    path.push(id.trim_start_matches('/'));
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(&path)
        .map_err(|e| ShmemError::MapOpenFailed(e.to_string()))?;
    let mmap_data = unsafe {
        MmapMut::map_mut(&file).map_err(|e| ShmemError::MapOpenFailed(e.to_string()))?
    };
    Ok(UnixMapData {
        file,
        mmap_data,
        map_size: size,
        unique_id: id.to_string(),
        is_owner: false,
    })
}

impl UnixMapData {
    pub fn set_owner(&mut self, owner: bool) {
        self.is_owner = owner;
    }

    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.mmap_data.as_ptr() as *mut u8
    }
}
