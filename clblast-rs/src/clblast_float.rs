use crate::clblast::{CLBlastLayout, CLBlastTranspose, CLCommandQueue, CLEvent, CLMem, CLBlastStatusCode};

pub trait CLBlastFloat: Sized {
    fn gemm(
        layout: CLBlastLayout, a_transpose: CLBlastTranspose,
        b_transpose: CLBlastTranspose, m: usize, n: usize,
        k: usize, a_buffer: CLMem,
        a_offset: usize, a_ld: usize, b_buffer: CLMem,
        b_offset: usize, b_ld: usize, c_buffer: CLMem,
        c_offset: usize, c_ld: usize, queue: *mut CLCommandQueue,
        event: *mut CLEvent
    ) -> CLBlastStatusCode;
    fn dot(
        n: usize, dot_buffer: CLMem, dot_offset: usize,
        x_buffer: CLMem, x_offset: usize, x_inc: usize,
        y_buffer: CLMem, y_offset: usize, y_inc: usize,
        queue: *mut CLCommandQueue, event: *mut CLEvent
    ) -> CLBlastStatusCode;
}
impl CLBlastFloat for f32 {
    fn gemm(
        layout: CLBlastLayout, a_transpose: CLBlastTranspose,
        b_transpose: CLBlastTranspose, m: usize, n: usize,
        k: usize, a_buffer: CLMem,
        a_offset: usize, a_ld: usize, b_buffer: CLMem,
        b_offset: usize, b_ld: usize, c_buffer: CLMem,
        c_offset: usize, c_ld: usize, queue: *mut CLCommandQueue,
        event: *mut CLEvent
    ) -> CLBlastStatusCode {
        unsafe {
            crate::clblast::clblast_sgemm(
                layout, a_transpose, b_transpose, m, n,
                k, 1.0 as f32, a_buffer, a_offset, a_ld, b_buffer,
                b_offset, b_ld, 0.0 as f32, c_buffer, c_offset, c_ld,
                queue, event
            )   
        }
    }
    fn dot(
        n: usize, dot_buffer: CLMem, dot_offset: usize,
        x_buffer: CLMem, x_offset: usize, x_inc: usize,
        y_buffer: CLMem, y_offset: usize, y_inc: usize,
        queue: *mut CLCommandQueue, event: *mut CLEvent
    ) -> CLBlastStatusCode {
        unsafe {
            crate::clblast::clblast_sdot(
                n, dot_buffer, dot_offset,
                x_buffer, x_offset, x_inc,
                y_buffer, y_offset, y_inc,
                queue, event
            )
        }
    }
}
impl CLBlastFloat for f64 {
    fn gemm(
        layout: CLBlastLayout, a_transpose: CLBlastTranspose,
        b_transpose: CLBlastTranspose, m: usize, n: usize,
        k: usize, a_buffer: CLMem,
        a_offset: usize, a_ld: usize, b_buffer: CLMem,
        b_offset: usize, b_ld: usize, c_buffer: CLMem,
        c_offset: usize, c_ld: usize, queue: *mut CLCommandQueue,
        event: *mut CLEvent
    ) -> CLBlastStatusCode {
        unsafe {
            crate::clblast::clblast_dgemm(
                layout, a_transpose, b_transpose, m, n,
                k, 1.0 as f64, a_buffer, a_offset, a_ld, b_buffer,
                b_offset, b_ld, 0.0 as f64, c_buffer, c_offset, c_ld,
                queue, event
            )
        }
    }
    fn dot(
        n: usize, dot_buffer: CLMem, dot_offset: usize,
        x_buffer: CLMem, x_offset: usize, x_inc: usize,
        y_buffer: CLMem, y_offset: usize, y_inc: usize,
        queue: *mut CLCommandQueue, event: *mut CLEvent
    ) -> CLBlastStatusCode {
        unsafe {
            crate::clblast::clblast_ddot(
                n, dot_buffer, dot_offset,
                x_buffer, x_offset, x_inc,
                y_buffer, y_offset, y_inc,
                queue, event
            )
        }
    }
}