use keter::lang::types::vector::Vec3;
use keter::prelude::*;
use keter_testbed::{App, MouseButton};

fn main() {
    let app = App::new("Gamma Test", [2048, 2048])
        .scale(1)
        .dpi(2.0)
        .gamma(1.0)
        .init();
    let display_kernel = DEVICE.create_kernel::<fn(bool, f32)>(&track!(|fr, gamma| {
        let value = if fr {
            0.5_f32.expr().powf(gamma)
        } else {
            ((dispatch_id().xy() / 2).reduce_sum() % 2).cast_f32()
        };
        app.display()
            .write(dispatch_id().xy(), Vec3::splat_expr(value));
    }));

    let mut gamma = 2.2;
    let mut b = false;

    println!("{:?}", 1.0 / 0.45);

    app.run(|rt, scope| {
        if rt.just_pressed_button(MouseButton::Left) {
            gamma -= 0.05;
            println!("{:?}", gamma);
        } else if rt.just_pressed_button(MouseButton::Right) {
            gamma += 0.05;
            println!("{:?}", gamma);
        } else if rt.just_pressed_button(MouseButton::Middle) {
            b = !b;
        }
        scope.submit([display_kernel.dispatch_async([2048, 2048, 1], &b, &gamma)]);
    });
}
