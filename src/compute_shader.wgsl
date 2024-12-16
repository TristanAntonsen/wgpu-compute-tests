@group(0) @binding(0) var texture_out: texture_storage_2d<rgba32float, write>; // For writing
@group(0) @binding(1) var<uniform> resolution: vec2<u32>;
@group(0) @binding(2) var<uniform> uniforms: vec4<u32>;

@compute
@workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {

    let pos = vec2(id.x, id.y);
    var fragColor: vec4<f32>;

    var current_color = vec4f(0.);

    // Determine the new color based on the current time
    var new_color = current_color; // Default to current sample
    
    let fres = vec2f(resolution);

    var p = vec2f(pos);
    var uv = (vec2f(p.x, fres.y - p.y) - 0.5*fres.xy)/min(fres.x, fres.y);

    fragColor = reflection_demo(uv);

    new_color = current_color + fragColor;

    // Write to output texture
    textureStore(texture_out, pos, new_color);
}


const FOCUS_DIST = 2.;

fn reflection_demo(uv: vec2f) -> vec4<f32> {

    // Ray Marching
    let rt = vec3<f32>(0., 0.375, 0.);
    var ro = vec3<f32>(0., 0.375, -2.0);

    let ray = ray_direction(uv, ro, rt);

    let hit = ray_march(ray);

    // Background 
    var v = length(uv) * .75;
    var fragColor = vec4<f32>(background(ray.rd), 1.);

    if (hit.distance <= 100.0) {
        let p = ray.ro + ray.rd * hit.distance;

        let color = ray_color_with_bounces(ray.ro, ray.rd, hit.color);
        // let color = simpleShading(p, ray.rd, hit.color);

        fragColor = vec4<f32>(color, 1.0);
    }

    return fragColor;
}


////////////////////////////////////////////////////////////////
// Constants
////////////////////////////////////////////////////////////////

const MAX_STEPS = 500;
const SURF_DIST: f32 = 0.0001;
const MAX_DIST: f32 = 100.0;
const PI: f32 = 3.141592653592;
const TAU: f32 = 6.283185307185;
const EPSILON: f32 =  0.001;
const LIPSCHITZ_FACTOR: f32 = 1.;
const WHITE = vec3<f32>(1.0);
const BLACK = vec3<f32>(0.);
const RED = vec3<f32>(1., 0., 0.);
const GREEN = vec3<f32>(0., 1., 0.);
const BLUE = vec3<f32>(0., 0., 1.);
const ORANGE = vec3<f32>(255.,95.,5.) / 255.;
const KELVIN2200 = vec3<f32>(255., 147., 44.) / 255.;
const BLUE2 = vec3<f32>(19.,41.,75.) / 255.;
const DEFAULT_MATERIAL = vec3<f32>(1., 0., 1.);
const MIRROR = vec3<f32>(0., 1., 1.);

////////////////////////////////////////////////////////////////
// Transformations
////////////////////////////////////////////////////////////////

fn Rot(a: f32) -> mat2x2f {
    let s = sin(a);
    let c = cos(a);
    return mat2x2f(c, -s, s, c);
}

fn rotX(p: vec3<f32>, a: f32) -> vec3<f32> {
    let s = sin(a);
    let c = cos(a);
    let m = mat3x3f(
        1., 0., 0.,
        0., c, -s,
        0., s, c,
        );
    return m * p;
}

fn rotY(p: vec3<f32>, a: f32) -> vec3<f32> {
    let s = sin(a);
    let c = cos(a);
    let m = mat3x3f(
        c, 0., s,
        0., 1., 0.,
        -s, 0., c,
        );
    return m * p;
}

fn rotZ(p: vec3<f32>, a: f32) -> vec3<f32> {
    let s = sin(a);
    let c = cos(a);
    let m = mat3x3f(
        c, -s, 0.,
        s,  c, 0.,
        0., 0., 1.
        );
    return m * p;
}

////////////////////////////////////////////////////////////////
// SDF Operations
////////////////////////////////////////////////////////////////

struct SDF {
    distance: f32,
    color: vec3<f32>,
    material: vec3<f32> // vec3f(roughness, metallic, tbd)
}


fn add(sdf: SDF, value: f32) -> SDF {
    return SDF(sdf.distance + value, sdf.color, sdf.material);
}

fn shell(sdf: SDF, value: f32) -> SDF {
    let d = abs(sdf.distance) - value / 2.;
    return SDF(d, sdf.color, sdf.material);
}

fn negate(sdf: SDF) -> SDF {
    return SDF(-sdf.distance, sdf.color, sdf.material);
}

fn opUnion(sdf1: SDF, sdf2: SDF ) -> SDF { 

    let d = min(sdf1.distance, sdf2.distance);
    var color = sdf1.color;
    var material = sdf1.material;
    if (sdf2.distance < sdf1.distance) { 
        color = sdf2.color; 
        material = sdf2.material;
        };

    return SDF(d, color, material); 

}

fn opSubtraction(sdf1: SDF, sdf2: SDF ) -> SDF { 

    let d = max(-sdf2.distance, sdf1.distance);
    var color = sdf1.color;
    var material = sdf1.material;
    if (sdf1.distance < d) { 
        color = sdf2.color; 
        material = sdf2.material;
        };

    return SDF(d, color, material); 

}

fn opIntersection(sdf1: SDF, sdf2: SDF ) -> SDF { 

    let d = max(sdf1.distance, sdf2.distance);
    var color = sdf1.color;
    var material = sdf1.material;
    if (sdf1.distance < sdf2.distance) { 
        color = sdf2.color; 
        material = sdf2.material;
        };

    return SDF(d, color, material); 

}

fn opSmoothUnion(sdf1: SDF, sdf2: SDF, k: f32 ) -> SDF { 

    let h = clamp( 0.5 + 0.5*(sdf2.distance-sdf1.distance)/k, 0.0, 1.0 );
    let d = mix( sdf2.distance, sdf1.distance, h ) - k*h*(1.0-h);
    var color = mix(sdf2.color, sdf1.color, h);
    var material = mix(sdf2.material, sdf1.material, h);

    return SDF(d, color, material); 
}

fn opSmoothSubtraction(sdf1: SDF, sdf2: SDF, k: f32 ) -> SDF { 

    let h = clamp( 0.5 - 0.5*(sdf2.distance+sdf1.distance)/k, 0.0, 1.0 );
    let d = mix( sdf1.distance, -sdf2.distance, h ) + k*h*(1.0-h);
    var color = mix(sdf1.color, sdf2.color, h);
    var material = mix(sdf1.material, sdf2.material, h);

    return SDF(d, color, material); 

}

fn opSmoothIntersection(sdf1: SDF, sdf2: SDF, k: f32 ) -> SDF { 

    let h = clamp( 0.5 - 0.5*(sdf2.distance-sdf1.distance)/k, 0.0, 1.0 );
    let d = mix( sdf2.distance, sdf1.distance, h ) + k*h*(1.0-h);
    var color = sdf1.color;
    var material = sdf1.material;
    if (sdf1.distance > sdf2.distance) {
        color = sdf2.color; 
        material = sdf2.material;
    };
    return SDF(d, color, material); 

}

////////////////////////////////////////////////////////////////
// Signed Distance Functions
////////////////////////////////////////////////////////////////
fn sdSphere(p: vec3<f32>, r: f32, color: vec3<f32>) -> SDF
{
    return SDF(length(p) - r, color, DEFAULT_MATERIAL);
}

fn sdBox( p: vec3<f32>, b: vec3<f32>, color: vec3<f32> ) -> SDF
{
    let q = abs(p) - b * 0.5;
    let d = length(max(q,vec3<f32>(0.0))) + min(max(q.x,max(q.y,q.z)),0.0);
    return SDF(d, color, DEFAULT_MATERIAL);
}

fn sdRoundBox( p: vec3<f32>, b: vec3<f32>, r: f32, color: vec3<f32> ) -> SDF
{
    let q = abs(p) - ((b * 0.5) - vec3<f32>(r));
    let d = length(max(q,vec3<f32>(0.0))) + min(max(q.x,max(q.y,q.z)),0.0) - r;
    return SDF(d, color, DEFAULT_MATERIAL);
}

fn sdPlane( p: vec3<f32>, n: vec3<f32>, h: f32, color: vec3<f32>) -> SDF {
  return SDF(dot(p,normalize(n)) + h, color, DEFAULT_MATERIAL);
}

fn sdRoundedCylinder( p: vec3<f32>, ra: f32, rb: f32, h: f32, color: vec3<f32>) -> SDF {
    let d0 = vec2f( length(p.xz)-2.0*ra+rb, abs(p.y) - h );
    let d = min(max(d0.x,d0.y),0.0) + length(vec2(max(d0.x,0.0), max(d0.y, 0.))) - rb; 
    return SDF(d, color, DEFAULT_MATERIAL);
}

fn sdGyroid(p: vec3<f32>, f: f32, color: vec3<f32>) -> SDF {
    // f: frequency

    let g = sin(f * p.x) * cos(f * p.y)
        + sin(f * p.y) * cos(f * p.z)
        + sin(f * p.z) * cos(f * p.x);

    return SDF(g / f, color, DEFAULT_MATERIAL);
}


////////////////////////////////////////////////////////////////
// Other Math
////////////////////////////////////////////////////////////////

fn two_body_field(a: SDF, b: SDF) -> f32 {
    return (a.distance - b.distance) / (a.distance + b.distance);
}

////////////////////////////////////////////////////////////////
// Main scene
////////////////////////////////////////////////////////////////

fn map(p: vec3<f32>) -> SDF {

    var d = cornell_box(p);
    return d;
}


fn cornell_box(p: vec3<f32>) -> SDF {
    let t = 0.01;
    let w = 0.7;
    let h = 0.7;
    let l = 1.8;
    let hw = w / 2.;
    let hh = h / 2.;
    
    var left = sdBox(p-vec3f(-hw, hh, 0.), vec3f(t, h, l), RED);
    var right = sdBox(p-vec3f(hw, hh, 0.), vec3f(t, h, l), GREEN);
    var top = sdBox(p-vec3f(0., h, 0.), vec3f(w * 2., t, l), WHITE);
    var bottom = sdBox(p-vec3f(0., -t/2., 0.), vec3f(w * 2., t, l), WHITE);
    var back = sdBox(p-vec3f(0., hh, l / 2.), vec3f(w, h, t), WHITE);

    var lw = w / 2.;
    var light = sdBox(p-vec3f(0., h, 0.), vec3f(lw, t * 1.2, l * 0.5), WHITE);
    light.material[2] = 5.;

    var dist = opUnion(left, right);
    dist = opUnion(dist, top);
    dist = opUnion(dist, bottom);
    dist = opUnion(dist, back);
    dist = opUnion(dist, light);

    var bh = 0.25;
    var psc = rotY(p-vec3f(-0.05, 0., 0.), PI/ 3.);
    var s1 = sdSphere(psc - vec3f(bh / 2., bh, bh / 2.), 0.125, BLUE2);
    s1.material = vec3f(0.1, 1., 0.);

    dist = opUnion(dist, s1);
    var s2 = sdRoundBox(psc - vec3f(0., bh / 2., 0.), vec3f(bh), 0.01, BLUE2);

    var sc = opSmoothUnion(s1, s2, 0.05);


    dist = opUnion(dist, sc);

    return dist;
}


// tetrahedral finite difference method
fn gradient(p: vec3<f32>) -> vec3<f32> {
    let k = vec2f(1,-1);
    return normalize( k.xyy*map( p + k.xyy*EPSILON ).distance + 
                      k.yyx*map( p + k.yyx*EPSILON ).distance + 
                      k.yxy*map( p + k.yxy*EPSILON ).distance + 
                      k.xxx*map( p + k.xxx*EPSILON ).distance );
}


////////////////////////////////////////////////////////////////
// Ray Marching Functions
////////////////////////////////////////////////////////////////

struct Ray {
    ro: vec3<f32>,
    rd: vec3<f32>
}


fn ray_march(ray: Ray) -> SDF {
    var d = 0.0;
    var color = WHITE;
    var material = DEFAULT_MATERIAL;

    var i: i32 = 0;
    loop {
        if i >= MAX_STEPS { break; }
        let p = ray.ro + ray.rd * d;
        let ds = map(p);
        d += ds.distance * LIPSCHITZ_FACTOR;
        color = ds.color;
        material = ds.material;
        if d >= MAX_DIST || ds.distance < SURF_DIST { break; }
        i++;
    }
    return SDF(d, color, material);
}

fn ray_color_with_bounces(ro_start: vec3<f32>, rd_start: vec3<f32>, color: vec3<f32>) -> vec3<f32> {
    let attenuation: f32 = 0.5;
    var i = 0;
    var MAX_BOUNCES = 8;
    var ray = Ray(ro_start, rd_start);
    var col = WHITE;
    
    loop {
        if i > MAX_BOUNCES { break; }
            let hit = ray_march(ray);
    
            if (hit.distance <= MAX_DIST) {

                let p = ray.ro + ray.rd * hit.distance;
                let normal = gradient(p);

                ray.ro = p + normal * EPSILON;


                ray.rd = reflect(ray.rd, normal);

                if hit.material[2] > 1. { 
                    col *= hit.color * hit.material[2];
                    break;
                } else {
                    col *= hit.color * attenuation;
                }

            } else {
                col *= background(ray.rd);
            }
            i++;
    }

    return col;

}

fn background(rd: vec3<f32>) -> vec3<f32> {
    let t = 0.5 * (rd.y + 1.0);
    // return mix(vec3(0.5, 0.7, 1.0), WHITE, t);

    return BLACK;
}

fn ray_direction(uv: vec2f, ro: vec3<f32>, rt: vec3<f32>) -> Ray {

    // screen orientation
    let vup = vec3<f32>(0., 1.0, 0.0);
    let aspectRatio = 1.; // taken care of elsewhere

    let vw = normalize(ro - rt);
    let vu = normalize(cross(vup, vw));
    let vv = cross(vw, vu);
    let theta = radians(30.); // half FOV
    let viewport_height = 2. * tan(theta);
    let viewport_width = aspectRatio * viewport_height;
    let horizontal = -viewport_width * vu;
    let vertical = viewport_height * vv;
    let center = ro - vw * FOCUS_DIST; // rt

    let point_screen_space = center + uv.x * horizontal + uv.y * vertical;
    let rd = point_screen_space - ro;

    let ray: Ray = Ray(ro, normalize(rd));

    return ray;
}



// For debugging
fn simpleShading(p: vec3<f32>, rd: vec3<f32>, albedo: vec3<f32>) -> vec3<f32> {
    let N = gradient(p);

    let light_pos_1 = vec3<f32>(8.0, 2.0, -8.0);
    let light_pos_2 = vec3<f32>(-2, .5, 2.);
    let light_pos_3 = vec3<f32>(0., 20.0, 0.);
    // let light_pos_4 = vec3<f32>(0., 40.0, 0.);
            
    // Lights
    let light1 = dot(N, normalize(light_pos_1))*.5+.5;
    let light2 = dot(N, normalize(light_pos_2))*.5+.5;
    let light3 = dot(N, normalize(light_pos_3))*.5+.5;
    let illumination = 0.33 * vec3(light1) + 0.33 * vec3(light2) + 0.33 * vec3(light3);;

    // Colors
    var color = albedo;
    color *= illumination; 

    return color;        
}