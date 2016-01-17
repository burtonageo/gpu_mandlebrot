//! Doc placeholder

extern crate opencl;
extern crate sdl2;

use sdl2::keyboard::Keycode;
use sdl2::pixels::{self, Color};
use sdl2::rect::Rect;

const KERNEL_SOURCE: &'static str = r"
__kernel void multiply_by_scalar(__global float const* const src_buf,
                                 __private float const coeff,
                                 __global float* dst_buf) {
    uint i = get_global_id(0);
    dst_buf[i] = src_buf[i] * coeff;
}
";

fn init_kernel() {
    let (device, ctx, queue) = opencl::util::create_compute_context()
                                   .expect("Could not initialise OpenCL");
    const COEFF: f32 = 5.4321;

    let vec_src = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

    let cl_src_buf = ctx.create_buffer::<f32>(vec_src.len(), opencl::cl::CL_MEM_READ_ONLY);
    let cl_dst_buf = ctx.create_buffer::<f32>(vec_src.len(), opencl::cl::CL_MEM_WRITE_ONLY);

    queue.write(&cl_src_buf, &&vec_src[..], ());

    let program = {
        let p = ctx.create_program_from_source(KERNEL_SOURCE);
        p.build(&device).expect("Could not build program");
        p
    };

    let kernel = program.create_kernel("multiply_by_scalar");
    kernel.set_arg(0, &cl_src_buf);
    kernel.set_arg(1, &COEFF);
    kernel.set_arg(2, &cl_dst_buf);

    let event = queue.enqueue_async_kernel(&kernel, vec_src.len(), None, ());

    let vec_dst: Vec<f32> = queue.get(&cl_dst_buf, &event);

    println!("  {:?}", &vec_src[..]);
    println!("* {:?}", COEFF);
    println!("= {:?}", &vec_dst[..]);
}

fn main() {
    init_kernel();

    let sdl = sdl2::init().expect("Could not intialise SDL library");
    let mut renderer = sdl.video()
                          .expect("Could not initialise video")
                          .window("mandlebrot", 800, 600)
                          .position_centered()
                          .build()
                          .expect("Could not create a window from the video subsystem")
                          .renderer()
                          .accelerated()
                          .present_vsync()
                          .target_texture()
                          .build()
                          .expect("Could not get renderer from window");

    renderer.set_draw_color(Color::RGB(255, 0, 0));

    let sz = renderer.window().map(|w| w.size()).unwrap_or((0, 0));
    let mut tex = renderer.create_texture_streaming(pixels::PixelFormatEnum::RGB888, sz).unwrap();

    let mut events = sdl.event_pump().expect("Could not initialise events");

    'mainloop: loop {
        for e in events.poll_iter() {
            use sdl2::event::Event;
            match e {
                Event::Quit {..} |
                Event::KeyDown {keycode: Some(Keycode::Escape), ..} => break 'mainloop,
                _ => {}
            }
        }
        renderer.clear();

        tex.with_lock(None, |pixels, pitch| {
               let (mut row, mut col) = (1, 1);
               for pixel in pixels {
                   if col >= pitch {
                       row += 1;
                       col = 1;
                   } else {
                       col += 1;
                   }
                   *pixel = 0x00 + col as u8;
               }
           })
           .expect("Could not write pixels to screen texture");

        renderer.copy(&tex, None, Some(Rect::new_unwrap(0, 0, 800, 600)));

        renderer.present();
    }
}
