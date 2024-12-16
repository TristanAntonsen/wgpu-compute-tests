use std::fs;
use std::time::Instant;
use image::Frames;
use serde_json::de;
use wgpu::util::DeviceExt;
use wgpu::Adapter;

const GRID_SIZE: u32 = 16;
const RESOLUTION: u32 = 1024;
const FRAMES: u32 = 1024;

pub fn main() {
    pollster::block_on(run());
}

pub struct ComputeState {
    device: wgpu::Device,
    queue: wgpu::Queue,
    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group: wgpu::BindGroup,
    output_texture: wgpu::Texture,
    uniform_buf: wgpu::Buffer,
    time: [u32; 4],
    last_frame_time: Instant,
}

impl ComputeState {
    async fn new(adapter: Adapter) -> ComputeState {
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    label: None,
                    memory_hints: Default::default(),
                },
                None, // Trace path
            )
            .await
            .unwrap();

        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(
                fs::read_to_string("src/compute_shader.wgsl")
                    .expect("Could not load shader")
                    .into(),
            ),
        });

        let output_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Compute Output Texture"),
            size: wgpu::Extent3d {
                width: RESOLUTION as u32,
                height: RESOLUTION as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let size_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Resolution Buffer"),
            contents: bytemuck::cast_slice(&[
                RESOLUTION,
                RESOLUTION,
            ]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[0, 0, 0, 0]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create a view for the output texture
        let output_texture_view =
            output_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let compute_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    // Output texture (storage)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba32Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    // Resolution buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Time buffer
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &compute_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&output_texture_view), // Bind output texture
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &size_buf,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniform_buf,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        let compute_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&compute_bind_group_layout],
                push_constant_ranges: &[],
            });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&compute_pipeline_layout),
            module: &compute_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let time = [0, 0, 0, 0];
        let last_frame_time = Instant::now();

        Self {
            device,
            queue,
            compute_pipeline,
            compute_bind_group,
            output_texture,
            uniform_buf,
            time,
            last_frame_time,
        }
    }

    fn update(&mut self) -> f32{
        let fps = self.fps();

        self.time[0] += 1;
        self.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::cast_slice(&[self.time]));

        fps
    }

    async fn save_rgba32float_to_rgba8_image(
        &self,
        output_staging_buffer: &wgpu::Buffer,
        resolution: (u32, u32),
        output_file: &str,
    ) {
        // Map the buffer to access the data
        let buffer_slice = output_staging_buffer.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.receive().await.unwrap().unwrap();

        // Retrieve the raw data
        let data = buffer_slice.get_mapped_range();
        let texture_width = resolution.0 as usize;
        let texture_height = resolution.1 as usize;
        let bytes_per_row = ((texture_width * 16 + 255) / 256) * 256; // Adjust for 256-byte alignment

        // Convert the f32 data to u8
        let mut rgba_u8_data = Vec::with_capacity(texture_width * texture_height * 4);
        for row in 0..texture_height {
            let row_start = row * bytes_per_row;
            let row_end = row_start + texture_width * 16; // 16 bytes per pixel for RGBA32Float
            let row_data = &data[row_start..row_end];

            for chunk in row_data.chunks_exact(16) {
                let r = f32::from_ne_bytes(chunk[0..4].try_into().unwrap());
                let g = f32::from_ne_bytes(chunk[4..8].try_into().unwrap());
                let b = f32::from_ne_bytes(chunk[8..12].try_into().unwrap());
                let a = f32::from_ne_bytes(chunk[12..16].try_into().unwrap());

                // Normalize and convert to u8
                rgba_u8_data.push((linear_to_gamma(r).clamp(0.0, 1.0) * 255.0) as u8);
                rgba_u8_data.push((linear_to_gamma(g).clamp(0.0, 1.0) * 255.0) as u8);
                rgba_u8_data.push((linear_to_gamma(b).clamp(0.0, 1.0) * 255.0) as u8);
                rgba_u8_data.push((linear_to_gamma(a).clamp(0.0, 1.0) * 255.0) as u8);
            }
        }
        // Save to an image
        use image::{ImageBuffer, Rgba};
        let image = ImageBuffer::<Rgba<u8>, _>::from_raw(
            RESOLUTION,
            RESOLUTION,
            rgba_u8_data,
        )
        .expect("Failed to create image buffer");
        image.save(output_file).expect("Failed to save image");

        // Unmap the buffer
        drop(data); // Ensure no references remain to the data slice
        output_staging_buffer.unmap();
    }

    fn fps(&mut self) -> f32 {
        // Calculate frame time
        let now = Instant::now();
        let frame_time = now - self.last_frame_time;
        self.last_frame_time = now;

        // if self.time[0] != 0 && self.time[0] < SAMPLES {
        let dt: f32 = frame_time.as_millis() as f32;
        let fps = (1000. / dt.max(0.001)).min(200.);
        return fps;
    }

    fn render(&mut self) {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            compute_pass.set_bind_group(0, &self.compute_bind_group, &[]);
            compute_pass.set_pipeline(&self.compute_pipeline);

            let workgroups_x = RESOLUTION / GRID_SIZE;
            let workgroups_y = RESOLUTION / GRID_SIZE;

            compute_pass.dispatch_workgroups(workgroups_x as u32, workgroups_y as u32, 1);
        }

        // self.device.poll(wgpu::Maintain::Wait);
        self.device.poll(wgpu::Maintain::Poll);
        // submit will accept anything that implements IntoIter
        self.queue.submit(std::iter::once(encoder.finish()));
    }
}

pub async fn run() {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();
    let mut compute_state = ComputeState::new(adapter).await;

    let mut debug_count = 0;

    for _ in 0..FRAMES {
        let fps = compute_state.update();
        if fps >= 200. {
            debug_count += 1;
        }
        println!("{:?}", fps);
        compute_state.render();
    }
    println!("FINISHED RENDER");
    println!("{} / {}", debug_count , FRAMES);
    finish_and_export(compute_state);
    println!("SAVED IMAGE");
}

pub fn finish_and_export(compute_state: ComputeState) {
    let mut encoder =
        compute_state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Copy Command Encoder"),
            });

    let output_staging_buffer = compute_state.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (RESOLUTION as u64
            * RESOLUTION as u64
            * 4
            * std::mem::size_of::<f32>() as u64),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    encoder.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &compute_state.output_texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &output_staging_buffer,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                // This needs to be padded to 256.
                bytes_per_row: Some((RESOLUTION * 16) as u32),
                rows_per_image: Some(RESOLUTION as u32),
            },
        },
        wgpu::Extent3d {
            width: RESOLUTION as u32,
            height: RESOLUTION as u32,
            depth_or_array_layers: 1,
        },
    );
    compute_state
        .queue
        .submit(std::iter::once(encoder.finish()));
    println!("SAVING");
    // Save the texture to an image before exiting
    pollster::block_on(async {
        compute_state
            .save_rgba32float_to_rgba8_image(
                &output_staging_buffer,
                (
                    RESOLUTION,
                    RESOLUTION,
                ),
                "output_image_compute.png",
            )
            .await;
    });
}

fn linear_to_gamma(linear: f32) -> f32 {
    if linear > 0. {
        return linear.sqrt();
    };

    return 0.;
}
