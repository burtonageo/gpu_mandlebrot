extern crate opencl;
extern crate sdl2;

use opencl::mem::CLBuffer;
use sdl2::pixels::Color;
use sdl2::keyboard::Keycode;

const KERNEL: &'static str = r"
__kernel void multiply_by_scalar(__global float const* const src_buf,
                                 __private float const coeff,
                                 __global float* dst_buf) {
    uint i = get_global_id(0);
    dst_buf[i] = src_buf[i] * coeff;
}
";

fn main() {
    let (device, ctx, queue) = opencl::util::create_compute_context().expect("Could not initialise OpenCL");

    let sdl = sdl2::init().expect("Could not intialise SDL library");
    let mut renderer = sdl.video()
                          .expect("Could not initialise video")
                          .window("mandlebrot", 800, 600)
                          .position_centered()
                          .build()
                          .expect("Could not create a window from the video subsystem")
                          .renderer()
                          .build()
                          .expect("Could not get renderer from window");

    let mut events = sdl.event_pump().expect("Could not initialise events");
    'mainloop: loop {
        for e in events.poll_iter() {
            use sdl2::event::Event;
            match e {
                Event::Quit {..} | Event::KeyDown {keycode: Some(Keycode::Escape), ..} => break 'mainloop,
                _ => { }
            }
        }

        renderer.set_draw_color(Color::RGB(255, 0, 0));
        renderer.clear();
        renderer.present();
    }
}