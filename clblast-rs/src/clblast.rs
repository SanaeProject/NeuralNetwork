use opencl3::memory::{Buffer, ClMem};

type CLBlastStatusCode = i32;
type CLBlastLayout = u32;
type CLBlastTranspose = u32;

type CLCommandQueue = *mut std::ffi::c_void;
type CLEvent = *mut std::ffi::c_void;
type CLMem = *mut std::ffi::c_void;

const DEFAULT_PLATFORM_ID: usize = 0;
const DEFAULT_DEVICE_ID: usize = 0;

pub const A_BUFFER: usize = 0;
pub const B_BUFFER: usize = 1;
pub const C_BUFFER: usize = 2;

unsafe extern "C" {
    #[link_name = "CLBlastSgemm"]
    pub unsafe fn clblast_sgemm(
        layout: CLBlastLayout, a_transpose: CLBlastTranspose,
        b_transpose: CLBlastTranspose, m: usize, n: usize,
        k: usize, alpha: f32, a_buffer: CLMem,
        a_offset: usize, a_ld: usize, b_buffer: CLMem,
        b_offset: usize, b_ld: usize, beta: f32, c_buffer: CLMem,
        c_offset: usize, c_ld: usize, queue: *mut CLCommandQueue,
        event: *mut CLEvent
    ) -> CLBlastStatusCode;
    #[link_name = "CLBlastDgemm"]
    pub unsafe fn clblast_dgemm(
        layout: CLBlastLayout, a_transpose: CLBlastTranspose,
        b_transpose: CLBlastTranspose, m: usize, n: usize,
        k: usize, alpha: f64, a_buffer: CLMem,
        a_offset: usize, a_ld: usize, b_buffer: CLMem,
        b_offset: usize, b_ld: usize, beta: f64, c_buffer: CLMem,
        c_offset: usize, c_ld: usize, queue: *mut CLCommandQueue,
        event: *mut CLEvent
    ) -> CLBlastStatusCode;

    #[link_name = "CLBlastSaxpy"]
    pub unsafe fn clblast_saxpy(
        n: usize, alpha: f32, x_buffer: CLMem, x_offset: usize, x_inc: usize,
        y_buffer: CLMem, y_offset: usize, y_inc: usize, queue: *mut CLCommandQueue,
        event: *mut CLEvent
    ) -> CLBlastStatusCode;
    #[link_name = "CLBlastDaxpy"]
    pub unsafe fn clblast_daxpy(
        n: usize, alpha: f64, x_buffer: CLMem, x_offset: usize, x_inc: usize,
        y_buffer: CLMem, y_offset: usize, y_inc: usize, queue: *mut CLCommandQueue,
        event: *mut CLEvent
    ) -> CLBlastStatusCode;
    
    #[link_name = "CLBlastSdot"]
    pub unsafe fn clblast_sdot(
        n: usize, dot_buffer: CLMem, dot_offset: usize,
        x_buffer: CLMem, x_offset: usize, x_inc: usize,
        y_buffer: CLMem, y_offset: usize, y_inc: usize,
        queue: *mut CLCommandQueue, event: *mut CLEvent
    ) -> CLBlastStatusCode;
    #[link_name = "CLBlastDdot"]
    pub unsafe fn clblast_ddot(
        n: usize, dot_buffer: CLMem, dot_offset: usize,
        x_buffer: CLMem, x_offset: usize, x_inc: usize,
        y_buffer: CLMem, y_offset: usize, y_inc: usize,
        queue: *mut CLCommandQueue, event: *mut CLEvent
    ) -> CLBlastStatusCode;
}

struct CLBlastMatrix<T> {
    buffer: Buffer<T>,
    row_major: bool,
    rows: usize,
    cols: usize,
}
pub struct CLBlast<T> {
    queue: opencl3::command_queue::CommandQueue,
    context: opencl3::context::Context,
    buffers: [Option<CLBlastMatrix<T>>; 3],
    ty: std::marker::PhantomData<T>,
}
impl<T> CLBlast<T> {
    pub fn new(platform_id: usize, device_id: usize) -> Result<CLBlast<T>, String> {
        let platforms = opencl3::platform::get_platforms()?;
        if platforms.len() <= platform_id { return Err(String::from("Platform ID is out of range")); }
        let platform = platforms[platform_id];

        let devices = platform.get_devices(opencl3::device::CL_DEVICE_TYPE_GPU)?;
        if devices.len() <= device_id { return Err(String::from("Device ID is out of range")); }
        let device = opencl3::device::Device::new(devices[device_id]);
        let context = opencl3::context::Context::from_device(&device)?;
        let queue = unsafe{ opencl3::command_queue::CommandQueue::create(&context, device.id(), opencl3::command_queue::CL_QUEUE_PROFILING_ENABLE)? };

        Ok(
            Self {
                queue: queue, 
                context: context, 
                buffers: [None, None, None],
                ty: std::marker::PhantomData 
            }
        )
    }
    pub fn set_buffer(&mut self, target: usize, vec: &[T], row_major: bool, rows: usize, cols: usize) -> Result<(), String>{
        if target >= self.buffers.len() {
            return Err(String::from("Target index is out of range"));
        }
        let mut buffer = unsafe {
            opencl3::memory::Buffer::<T>::create(
                &self.context, opencl3::memory::CL_MEM_WRITE_ONLY, vec.len(), std::ptr::null_mut()
            )?
        };
        unsafe{
            self.queue.enqueue_write_buffer(&mut buffer, opencl3::types::CL_TRUE, 0, &vec[..], &[])?;
        }

        self.buffers[target] = Some(CLBlastMatrix { buffer: buffer, row_major: row_major, rows: rows, cols: cols });
        Ok(())
    }
    pub fn swap_buffer(&mut self, target1: usize, target2: usize) -> Result<(), String> {
        if target1 >= self.buffers.len() || target2 >= self.buffers.len() {
            return Err(String::from("Target1 index is out of range"));
        }
        if target1 == target2 {
            return Ok(());
        }

        self.buffers.swap(target1, target2);
        Ok(())
    }
    pub fn read_buffer(&mut self, target: usize, vec: &mut [T]) -> Result<(), String> {
        let buf = self.buffers[target].as_ref().ok_or_else(|| String::from("Buffer not set"))?;

        unsafe{
            self.queue.enqueue_read_buffer(&buf.buffer, opencl3::types::CL_TRUE, 0, vec, &[])?;
        }
        Ok(())
    }
}

pub enum Layout {
    RowMajor = 101,
    ColMajor = 102,
}
pub enum Transpose {
    No = 111,
    Yes = 112,
}

impl CLBlast<f32> {
    pub fn mat_mul(&mut self) -> Result<(), String> {
        let a = self.buffers[A_BUFFER].as_ref().ok_or_else(|| String::from("Buffer A not set"))?;
        let b = self.buffers[B_BUFFER].as_ref().ok_or_else(|| String::from("Buffer B not set"))?;
        let c = self.buffers[C_BUFFER].as_ref().ok_or_else(|| String::from("Buffer C not set"))?;

        let layout = if a.row_major { Layout::RowMajor } else { Layout::ColMajor };
        
        let (a_ld, b_ld, c_ld) = if a.row_major {
            (a.cols, b.cols, c.cols)
        } else {
            (a.rows, b.rows, c.rows)
        };
        let mut raw_queue = self.queue.get();

        let status = unsafe {
            clblast_sgemm(
                layout as CLBlastLayout,
                Transpose::No as CLBlastTranspose,
                Transpose::No as CLBlastTranspose,
                a.rows, // m
                b.cols, // n
                a.cols, // k
                1.0,
                a.buffer.get(), 0, a_ld,
                b.buffer.get(), 0, b_ld,
                0.0,
                c.buffer.get(), 0, c_ld,
                &mut raw_queue,
                std::ptr::null_mut(),
            )
        };

        if status == 0 {
            Ok(())
        } else {
            Err(format!("CLBlast GEMM failed with status code: {}", status))
        }
    }
}