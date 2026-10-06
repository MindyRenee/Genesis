//! Machine-native field architecture for Genesis.
//!
//! A major system is a SystemFiber; its subsystems are the layers of that
//! system's latent field. Geometry is separated from semantics: a manifold
//! is a container and metric, while layers own represented state and dynamics.
//!
//! The implementation is dependency-free and deterministic. It does not
//! assume that a fixed number of fibers is correct.

use std::fmt;

pub mod engine;
pub use engine::SystemEngine;
pub mod runtime;
pub use runtime::{FieldIntegration, FieldRuntime, LATENT_DIMENSION};

const EPS: f64 = 1.0e-12;
pub type Vector = Vec<f64>;
pub type Matrix = Vec<Vector>;

fn dot(a: &[f64], b: &[f64]) -> f64 {
    let scale_a = a.iter().fold(0.0_f64, |scale, &x| scale.max(x.abs()));
    let scale_b = b.iter().fold(0.0_f64, |scale, &x| scale.max(x.abs()));
    if scale_a == 0.0 || scale_b == 0.0 {
        return 0.0;
    }
    let normalized = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x / scale_a) * (y / scale_b))
        .sum::<f64>();
    if normalized == 0.0 {
        return 0.0;
    }
    (normalized * scale_a) * scale_b
}
fn norm(a: &[f64]) -> f64 { a.iter().fold(0.0_f64, |acc, &x| acc.hypot(x)) }
fn scaled(a: &[f64], s: f64) -> Vector { a.iter().map(|x| x * s).collect() }
fn add(a: &[f64], b: &[f64]) -> Vector { a.iter().zip(b).map(|(x, y)| x + y).collect() }
fn sub(a: &[f64], b: &[f64]) -> Vector { a.iter().zip(b).map(|(x, y)| x - y).collect() }

#[derive(Clone, Debug, PartialEq)]
pub struct PoincareBall { dimension: usize, curvature: f64 }

impl PoincareBall {
    pub fn new(dimension: usize, c: f64) -> Result<Self, FieldError> { if dimension == 0 || !c.is_finite() || c <= 0.0 { return Err(FieldError::InvalidGeometry); } Ok(Self { dimension, curvature: c }) }
    pub fn dimension(&self) -> usize { self.dimension }
    pub fn curvature(&self) -> f64 { self.curvature }
    fn scaled_radius(&self) -> f64 { 1.0 / self.curvature.sqrt() }
    pub fn contains(&self, x: &[f64]) -> bool { x.len() == self.dimension && x.iter().all(|v| v.is_finite()) && norm(x) < self.scaled_radius() }
    pub fn project(&self, x: &[f64]) -> Result<Vector, FieldError> {
        if x.len() != self.dimension {
            return Err(FieldError::DimensionMismatch);
        }
        if x.iter().any(|v| !v.is_finite()) {
            return Err(FieldError::NonFiniteState);
        }
        let r = self.scaled_radius();
        let n = norm(x);
        if !n.is_finite() {
            return Err(FieldError::NonFiniteState);
        }
        if n < r {
            return Ok(x.to_vec());
        }
        let target = r * (1.0 - 1.0e-9);
        if !target.is_finite() || target <= 0.0 {
            return Err(FieldError::InvalidGeometry);
        }
        let scale = target / n;
        if !scale.is_finite() || scale <= 0.0 {
            return Err(FieldError::NonFiniteState);
        }
        let projected = scaled(x, scale);
        if projected.iter().any(|v| !v.is_finite()) || !self.contains(&projected) {
            return Err(FieldError::NonFiniteState);
        }
        Ok(projected)
    }
    pub fn distance(&self,x:&[f64],y:&[f64])->Result<f64,FieldError>{self.validate_point(x)?;self.validate_point(y)?;let sqrt_c=self.curvature.sqrt();let ux=scaled(x,sqrt_c);let uy=scaled(y,sqrt_c);let nx=dot(&ux,&ux);let ny=dot(&uy,&uy);let dxy=norm(&sub(&ux,&uy));if !nx.is_finite()||!ny.is_finite()||!dxy.is_finite(){return Err(FieldError::NonFiniteState)}let denom=(1.0-nx).max(EPS)*(1.0-ny).max(EPS);let arg=1.0+2.0*dxy*dxy/denom;if !arg.is_finite(){return Err(FieldError::NonFiniteState)}let distance=arg.max(1.0).acosh()/sqrt_c;if !distance.is_finite(){return Err(FieldError::NonFiniteState)}Ok(distance)}
    pub fn log_origin(&self,x:&[f64])->Result<Vector,FieldError>{
        self.validate_point(x)?;
        let r = norm(x);
        if !r.is_finite() {
            return Err(FieldError::NonFiniteState);
        }
        if r < EPS {
            return Ok(vec![0.0; self.dimension]);
        }
        let sqrt_c = self.curvature.sqrt();
        let scaled_r = sqrt_c * r;
        if !scaled_r.is_finite() || scaled_r <= 0.0 {
            return Err(FieldError::NonFiniteState);
        }
        let atanh_arg = scaled_r.min(1.0 - f64::EPSILON);
        let z = atanh_arg.atanh() * 2.0 / scaled_r;
        if !z.is_finite() || z <= 0.0 {
            return Err(FieldError::NonFiniteState);
        }
        let tangent = scaled(x, z);
        if tangent.iter().any(|v| !v.is_finite()) {
            return Err(FieldError::NonFiniteState);
        }
        Ok(tangent)
    }
    pub fn exp_origin(&self,v:&[f64])->Result<Vector,FieldError>{
        if v.len()!=self.dimension {
            return Err(FieldError::DimensionMismatch);
        }
        if v.iter().any(|x|!x.is_finite()) {
            return Err(FieldError::NonFiniteState);
        }
        let r=norm(v);
        if !r.is_finite() {
            return Err(FieldError::NonFiniteState);
        }
        if r<EPS {
            return Ok(v.to_vec());
        }
        let sqrt_c=self.curvature.sqrt();
        let scaled_r=sqrt_c*r;
        let z=if scaled_r.is_finite(){
            (scaled_r/2.0).tanh()/scaled_r
        }else{
            (1.0/sqrt_c)/r
        };
        if !z.is_finite() || z <= 0.0 {
            return Err(FieldError::NonFiniteState);
        }
        self.project(&scaled(v,z))
    }
    fn validate_point(&self,x:&[f64])->Result<(),FieldError>{if self.contains(x){Ok(())}else{Err(FieldError::InvalidPoint)}}
    pub fn origin_barycenter(&self,points:&[Vector])->Result<Vector,FieldError>{
        if points.is_empty() {
            return Err(FieldError::EmptyState);
        }
        if points.iter().any(|p| p.len() != self.dimension) {
            return Err(FieldError::DimensionMismatch);
        }
        let tangents = points.iter()
            .map(|p| self.log_origin(p))
            .collect::<Result<Vec<_>, _>>()?;
        let scale = tangents.iter()
            .flatten()
            .fold(0.0_f64, |m, &v| m.max(v.abs()));
        if !scale.is_finite() {
            return Err(FieldError::NonFiniteState);
        }
        if scale == 0.0 {
            return self.exp_origin(&vec![0.0; self.dimension]);
        }
        let count = points.len() as f64;
        let mut mean = vec![0.0; self.dimension];
        for tangent in &tangents {
            for (dst, &src) in mean.iter_mut().zip(tangent) {
                *dst += (src / scale) / count;
            }
        }
        if mean.iter().any(|v| !v.is_finite()) {
            return Err(FieldError::NonFiniteState);
        }
        let mean = scaled(&mean, scale);
        if mean.iter().any(|v| !v.is_finite()) {
            return Err(FieldError::NonFiniteState);
        }
        self.exp_origin(&mean)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SubsystemLayer { pub id:String, pub state:Vector, pub gain:f64 }
impl SubsystemLayer { pub fn new(id:impl Into<String>,state:Vector)->Self{Self{id:id.into(),state,gain:1.0}} pub fn evolve(&mut self,derivative:&[f64],dt:f64)->Result<(),FieldError>{if derivative.len()!=self.state.len(){return Err(FieldError::DimensionMismatch)}if !dt.is_finite()||dt<0.0{return Err(FieldError::InvalidTime)}if !self.gain.is_finite(){return Err(FieldError::InvalidDynamics)}if derivative.iter().any(|v|!v.is_finite()){return Err(FieldError::NonFiniteState)}let mut next=self.state.clone();for(x,dx)in next.iter_mut().zip(derivative){*x+=dt*self.gain*dx;}if next.iter().any(|v|!v.is_finite()){return Err(FieldError::NonFiniteState)}self.state=next;Ok(())} }

#[derive(Clone, Debug, PartialEq)]
pub struct SystemFiber { pub id:String,pub engine_clock:f64,pub manifold:PoincareBall,pub layers:Vec<SubsystemLayer>,pub engine:SystemEngine }
impl SystemFiber {
 pub fn new(id:impl Into<String>,manifold:PoincareBall,layers:Vec<SubsystemLayer>)->Result<Self,FieldError>{
    if layers.iter().any(|l|l.state.len()!=manifold.dimension()){
        return Err(FieldError::DimensionMismatch);
    }
    if layers.iter().any(|l|l.state.iter().any(|v|!v.is_finite())){
        return Err(FieldError::NonFiniteState);
    }
    if layers.iter().any(|l|!l.gain.is_finite()){
        return Err(FieldError::InvalidDynamics);
    }
    let mut ids=std::collections::HashSet::with_capacity(layers.len());
    if layers.iter().any(|l|!ids.insert(&l.id)){
        return Err(FieldError::DuplicateId);
    }
    Ok(Self{id:id.into(),engine_clock:0.0,engine:SystemEngine::zero(layers.len(),manifold.dimension()),manifold,layers})
}
 pub fn with_engine(id:impl Into<String>,manifold:PoincareBall,layers:Vec<SubsystemLayer>,engine:SystemEngine)->Result<Self,FieldError>{if engine.biases.len()!=layers.len(){return Err(FieldError::LayerMismatch)}if engine.dimension()!=manifold.dimension(){return Err(FieldError::DimensionMismatch)}engine.validate()?;let mut fiber=Self::new(id,manifold,layers)?;fiber.engine=engine;Ok(fiber)}
 pub fn layer(&self,id:&str)->Option<&SubsystemLayer>{self.layers.iter().find(|l|l.id==id)} pub fn layer_mut(&mut self,id:&str)->Option<&mut SubsystemLayer>{self.layers.iter_mut().find(|l|l.id==id)}
 pub fn step(&mut self,dt:f64)->Result<(),FieldError>{if !self.engine_clock.is_finite()||self.engine_clock<0.0{return Err(FieldError::InvalidTime)}if !dt.is_finite()||dt<0.0{return Err(FieldError::InvalidTime)}self.engine.validate()?;if self.layers.iter().any(|l|l.state.iter().any(|v|!v.is_finite())){return Err(FieldError::NonFiniteState)}let effective_dt=dt*self.engine.time_scale;if !effective_dt.is_finite()||effective_dt<0.0{return Err(FieldError::InvalidTime)}if effective_dt==0.0{return Ok(())}if self.layers.iter().any(|l|!l.gain.is_finite()){return Err(FieldError::InvalidDynamics)}let initial=self.layer_states();let scaled_derivatives=|ds:Vec<Vector>|->Vec<Vector>{ds.into_iter().enumerate().map(|(i,d)|d.into_iter().map(|x|x*self.layers[i].gain).collect()).collect()};let add_stage=|state:&[Vector],slope:&[Vector],scale:f64|->Vec<Vector>{state.iter().zip(slope).map(|(x,dx)|x.iter().zip(dx).map(|(v,dv)|v+scale*dv).collect()).collect()};let k1=scaled_derivatives(self.engine.derivatives(&self.layers)?);let state2=add_stage(&initial,&k1,effective_dt*0.5);let mut stage_layers=self.layers.clone();for(layer,state)in stage_layers.iter_mut().zip(state2){layer.state=state}let k2=scaled_derivatives(self.engine.derivatives(&stage_layers)?);let state3=add_stage(&initial,&k2,effective_dt*0.5);for(layer,state)in stage_layers.iter_mut().zip(state3){layer.state=state}let k3=scaled_derivatives(self.engine.derivatives(&stage_layers)?);let state4=add_stage(&initial,&k3,effective_dt);for(layer,state)in stage_layers.iter_mut().zip(state4){layer.state=state}let k4=scaled_derivatives(self.engine.derivatives(&stage_layers)?);let mut next_layers=self.layers.clone();for(i,layer)in next_layers.iter_mut().enumerate(){for j in 0..layer.state.len(){layer.state[j]=initial[i][j]+effective_dt*(k1[i][j]+2.0*k2[i][j]+2.0*k3[i][j]+k4[i][j])/6.0;}layer.state=self.manifold.project(&layer.state)?;}let next_clock=self.engine_clock+effective_dt;if !next_clock.is_finite(){return Err(FieldError::InvalidTime)}self.layers=next_layers;self.engine_clock=next_clock;Ok(())}
 pub fn evolve(&mut self,derivatives:&[Vector],dt:f64)->Result<(),FieldError>{if !self.engine_clock.is_finite()||self.engine_clock<0.0{return Err(FieldError::InvalidTime)}if !dt.is_finite()||dt<0.0{return Err(FieldError::InvalidTime)}if derivatives.len()!=self.layers.len(){return Err(FieldError::LayerMismatch)}self.engine.validate()?;if self.layers.iter().any(|l|l.state.iter().any(|v|!v.is_finite())){return Err(FieldError::NonFiniteState)}let mut next_layers=self.layers.clone();for(layer,derivative)in next_layers.iter_mut().zip(derivatives){layer.evolve(derivative,dt)?;layer.state=self.manifold.project(&layer.state)?;}let next_clock=self.engine_clock+dt;if !next_clock.is_finite(){return Err(FieldError::InvalidTime)}self.layers=next_layers;self.engine_clock=next_clock;Ok(())}
 pub fn state(&self)->Result<Vector,FieldError>{if self.layers.is_empty(){return Err(FieldError::EmptyState)}self.manifold.origin_barycenter(&self.layers.iter().map(|l|l.state.clone()).collect::<Vec<_>>())} pub fn layer_states(&self)->Vec<Vector>{self.layers.iter().map(|l|l.state.clone()).collect()}
}

#[derive(Clone, Debug, PartialEq)]
pub struct GaugeTransport {pub from:String,pub to:String,pub matrix:Matrix}
impl GaugeTransport{pub fn new(from:impl Into<String>,to:impl Into<String>,matrix:Matrix)->Result<Self,FieldError>{let n=matrix.len();if n==0||matrix.iter().any(|r|r.len()!=n||r.iter().any(|v|!v.is_finite())){return Err(FieldError::InvalidTransport)}Ok(Self{from:from.into(),to:to.into(),matrix})}pub fn apply_on_manifold(&self,source:&PoincareBall,target:&PoincareBall,x:&[f64])->Result<Vector,FieldError>{if source.dimension()!=target.dimension()||source.curvature()!=target.curvature()||self.matrix.len()!=source.dimension(){return Err(FieldError::GeometryMismatch)}let tangent=source.log_origin(x)?;target.exp_origin(&self.apply_tangent(&tangent)?)}pub fn apply_tangent(&self,x:&[f64])->Result<Vector,FieldError>{if x.len()!=self.matrix.len(){return Err(FieldError::DimensionMismatch)}let mut out=vec![0.0;x.len()];for(i,row)in self.matrix.iter().enumerate(){if row.len()!=x.len(){return Err(FieldError::DimensionMismatch)}out[i]=dot(row,x);}if out.iter().any(|v|!v.is_finite()){return Err(FieldError::NonFiniteState)}Ok(out)}pub fn compose(&self,other:&GaugeTransport)->Result<GaugeTransport,FieldError>{if self.from!=other.to||self.matrix.len()!=other.matrix.len(){return Err(FieldError::InvalidTransport)}let n=self.matrix.len();let mut m=vec![vec![0.0;n];n];for i in 0..n{for j in 0..n{for k in 0..n{let product=self.matrix[i][k]*other.matrix[k][j];if !product.is_finite(){return Err(FieldError::NonFiniteState)}let sum=m[i][j]+product;if !sum.is_finite(){return Err(FieldError::NonFiniteState)}m[i][j]=sum;}}}GaugeTransport::new(other.from.clone(),self.to.clone(),m)}}

#[derive(Clone,Debug,PartialEq)]
pub struct BraidTrajectory{pub fiber_id:String,pub samples:Vec<Vector>}
impl BraidTrajectory{pub fn new(fiber_id:impl Into<String>)->Self{Self{fiber_id:fiber_id.into(),samples:Vec::new()}}pub fn push(&mut self,state:Vector){self.samples.push(state)}pub fn len(&self)->usize{self.samples.len()}pub fn is_empty(&self)->bool{self.samples.is_empty()}}

#[derive(Clone,Debug,PartialEq)]
pub struct TransportPath{pub nodes:Vec<String>,pub matrix:Matrix}
#[derive(Clone,Debug,PartialEq)]
pub struct PathDisagreement{pub from:String,pub to:String,pub paths:Vec<TransportPath>,pub tangent_spread:f64}

#[derive(Clone,Debug,PartialEq)]
pub struct GlobalField{pub manifold:PoincareBall,pub fibers:Vec<SystemFiber>,pub transports:Vec<GaugeTransport>,pub integration_gain:f64,pub disagreement:f64,pub temperature:f64,pub cooling_rate:f64}
impl GlobalField{pub fn new(manifold:PoincareBall)->Self{Self{manifold,fibers:Vec::new(),transports:Vec::new(),integration_gain:1.0,disagreement:0.0,temperature:1.0,cooling_rate:0.1}}pub fn add_fiber(&mut self,fiber:SystemFiber)->Result<(),FieldError>{if fiber.manifold.dimension()!=self.manifold.dimension()||(fiber.manifold.curvature()-self.manifold.curvature()).abs()>EPS{return Err(FieldError::GeometryMismatch)}if fiber.layers.iter().any(|l|l.state.iter().any(|v|!v.is_finite())||!l.gain.is_finite()){return Err(FieldError::InvalidDynamics)}fiber.engine.validate()?;if !fiber.engine_clock.is_finite()||fiber.engine_clock<0.0{return Err(FieldError::InvalidTime)}if self.fibers.iter().any(|f|f.id==fiber.id){return Err(FieldError::DuplicateId)}self.fibers.push(fiber);Ok(())}pub fn add_transport(&mut self,transport:GaugeTransport)->Result<(),FieldError>{let n=self.manifold.dimension();if transport.from==transport.to||!self.fibers.iter().any(|f|f.id==transport.from)||!self.fibers.iter().any(|f|f.id==transport.to)||transport.matrix.len()!=n||transport.matrix.iter().any(|r|r.len()!=n||r.iter().any(|v|!v.is_finite())){return Err(FieldError::TransportMismatch)}self.transports.push(transport);Ok(())}fn transport_paths(&self,from:&str,to:&str,max_paths:usize)->Result<Vec<TransportPath>,FieldError>{fn walk(field:&GlobalField,node:&str,to:&str,matrix:Matrix,nodes:Vec<String>,seen:&mut Vec<String>,out:&mut Vec<TransportPath>,max_paths:usize)->Result<(),FieldError>{if out.len()>=max_paths{return Ok(())}if node==to{out.push(TransportPath{nodes,matrix});return Ok(())}for edge in field.transports.iter().filter(|e|e.from==node){if seen.iter().any(|id|id==&edge.to){continue}seen.push(edge.to.clone());let next=mat_mul(&edge.matrix,&matrix)?;let mut next_nodes=nodes.clone();next_nodes.push(edge.to.clone());walk(field,&edge.to,to,next,next_nodes,seen,out,max_paths)?;seen.pop();}Ok(())}let mut out=Vec::new();let mut seen=vec![from.to_string()];walk(self,from,to,identity(self.manifold.dimension()),vec![from.to_string()],&mut seen,&mut out,max_paths.max(1))?;Ok(out)}pub fn path_disagreement(&self,from:&str,to:&str)->Result<Option<PathDisagreement>,FieldError>{if from==to{return Err(FieldError::TransportMismatch)}let paths=self.transport_paths(from,to,32)?;if paths.len()<2{return Ok(None)}let mut spread: f64=0.0;for i in 0..paths.len(){for j in i+1..paths.len(){spread=spread.max(matrix_distance(&paths[i].matrix,&paths[j].matrix)?);}}if !spread.is_finite(){return Err(FieldError::NonFiniteState)}Ok(Some(PathDisagreement{from:from.into(),to:to.into(),paths,tangent_spread:spread}))}fn path_matrix(&self,from:&str,to:&str)->Result<Matrix,FieldError>{let paths=self.transport_paths(from,to,32)?;match paths.as_slice(){[]=>Err(FieldError::MissingTransport),[p]=>Ok(p.matrix.clone()),[first,rest @ ..]=>{let candidate=&first.matrix;for path in rest{if matrix_distance(candidate,&path.matrix)?>1.0e-9{return Err(FieldError::TransportMismatch)}}Ok(candidate.clone())}}}pub fn transported_states(&self)->Result<Vec<Vector>,FieldError>{if self.fibers.is_empty(){return Ok(Vec::new())}let reference=&self.fibers[0];let states=std::iter::once(Ok(reference.state()?)).chain(self.fibers.iter().skip(1).map(|fiber|{let state=fiber.state()?;let matrix=self.path_matrix(&fiber.id,&reference.id)?;let transport=GaugeTransport::new(fiber.id.clone(),reference.id.clone(),matrix)?;transport.apply_on_manifold(&fiber.manifold,&reference.manifold,&state)})).collect::<Result<Vec<_>,FieldError>>()?;if states.iter().any(|state|state.iter().any(|value|!value.is_finite())){return Err(FieldError::NonFiniteState)}Ok(states)}pub fn consensus(&self)->Result<Option<Vector>,FieldError>{let states=self.transported_states()?;if states.is_empty(){Ok(None)}else{Ok(Some(self.manifold.origin_barycenter(&states)?))}}pub fn step(&mut self,dt:f64)->Result<Option<Vector>,FieldError>{
    if !dt.is_finite()||dt<0.0{return Err(FieldError::InvalidTime)}
    let mut next=self.clone();
    for fiber in &mut next.fibers {
        fiber.step(dt)?;
    }
    let consensus=next.synchronize(dt)?;
    *self=next;
    Ok(consensus)
}pub fn synchronize(&mut self,dt:f64)->Result<Option<Vector>,FieldError>{if !dt.is_finite()||dt<0.0{return Err(FieldError::InvalidTime)}if !self.integration_gain.is_finite()||!self.temperature.is_finite()||!self.cooling_rate.is_finite()||self.integration_gain<0.0||self.temperature<0.0||self.cooling_rate<0.0{return Err(FieldError::InvalidDynamics)}let states=self.transported_states()?;if states.is_empty(){return Ok(None)}let consensus=self.manifold.origin_barycenter(&states)?;let mut disagreement=0.0;for state in &states{let d=self.manifold.distance(state,&consensus)?;disagreement+=d*d;}let next_disagreement=disagreement/states.len() as f64;let gain=(self.integration_gain.max(0.0)*dt).clamp(0.0,1.0);let reference_id=self.fibers[0].id.clone();let consensus_tangent=self.manifold.log_origin(&consensus)?;let mut next_states:Vec<(usize,Vec<Vector>)>=Vec::with_capacity(self.fibers.len().saturating_sub(1));for i in 1..self.fibers.len(){let matrix=self.path_matrix(&self.fibers[i].id,&reference_id)?;let inv=matrix_inverse(&matrix)?;let local_target_tangent=mat_vec(&inv,&consensus_tangent)?;let current=self.fibers[i].state()?;let current_tangent=self.fibers[i].manifold.log_origin(&current)?;let correction=scaled(&sub(&local_target_tangent,&current_tangent),gain);let manifold=self.fibers[i].manifold.clone();let mut states_for_fiber=Vec::with_capacity(self.fibers[i].layers.len());for layer in &self.fibers[i].layers{let tangent=manifold.log_origin(&layer.state)?;let next=manifold.exp_origin(&add(&tangent,&correction))?;states_for_fiber.push(next);}next_states.push((i,states_for_fiber));}let next_temperature=(self.temperature-self.cooling_rate*dt).max(0.0);self.disagreement=next_disagreement;for(i,states_for_fiber)in next_states{for(layer,next)in self.fibers[i].layers.iter_mut().zip(states_for_fiber){layer.state=next;}}self.temperature=next_temperature;Ok(Some(consensus))}}

fn identity(n:usize)->Matrix{let mut m=vec![vec![0.0;n];n];for i in 0..n{m[i][i]=1.0;}m}
fn mat_mul(a:&Matrix,b:&Matrix)->Result<Matrix,FieldError>{
    if a.is_empty()||b.is_empty()||a.len()!=b.len()
        ||a.iter().any(|row|row.len()!=a.len())
        ||b.iter().any(|row|row.len()!=b.len()){
        return Err(FieldError::DimensionMismatch);
    }
    let n=a.len();
    let mut out=vec![vec![0.0;n];n];
    for i in 0..n {
        for j in 0..n {
            out[i][j]=dot(&a[i], &(0..n).map(|k| b[k][j]).collect::<Vec<_>>());
            if !out[i][j].is_finite(){
                return Err(FieldError::NonFiniteState);
            }
        }
    }
    Ok(out)
}
fn mat_vec(a:&Matrix,x:&Vector)->Result<Vector,FieldError>{if a.len()!=x.len()||a.iter().any(|r|r.len()!=x.len()){return Err(FieldError::DimensionMismatch)}let mut out=vec![0.0;x.len()];for i in 0..a.len(){out[i]=dot(&a[i],x);}if out.iter().any(|v|!v.is_finite()){return Err(FieldError::NonFiniteState)}Ok(out)}
fn matrix_distance(a:&Matrix,b:&Matrix)->Result<f64,FieldError>{
    if a.len()!=b.len()||a.iter().zip(b).any(|(x,y)|x.len()!=y.len()){
        return Err(FieldError::DimensionMismatch);
    }
    let mut scale=0.0_f64;
    for (row_a,row_b) in a.iter().zip(b) {
        for (&x,&y) in row_a.iter().zip(row_b) {
            if !x.is_finite()||!y.is_finite(){return Err(FieldError::NonFiniteState);}
            let delta=x-y;
            if !delta.is_finite(){return Err(FieldError::NonFiniteState);}
            scale=scale.max(delta.abs());
        }
    }
    if scale==0.0{return Ok(0.0);}
    if !scale.is_finite(){return Err(FieldError::NonFiniteState);}
    let normalized_sum=a.iter().zip(b).flat_map(|(row_a,row_b)|row_a.iter().zip(row_b))
        .map(|(&x,&y)|{
            let delta=x-y;
            let normalized=delta/scale;
            normalized*normalized
        })
        .sum::<f64>();
    if !normalized_sum.is_finite(){return Err(FieldError::NonFiniteState);}
    let distance=scale*normalized_sum.sqrt();
    if !distance.is_finite(){return Err(FieldError::NonFiniteState);}
    Ok(distance)
}
#[cfg(test)]
mod vector_numerics_tests {
    use super::*;

    #[test]
    fn dot_handles_large_cancelling_products() {
        assert_eq!(dot(&[1.0e308, 1.0e308], &[1.0e308, -1.0e308]), 0.0);
    }
}

#[cfg(test)]
mod subsystem_layer_integrity_tests {
    use super::*;

    #[test]
    fn evolve_reports_invalid_time_separately_from_dimension_mismatch() {
        let mut layer = SubsystemLayer::new("x", vec![0.0, 0.0]);
        assert_eq!(layer.evolve(&[1.0, 1.0], f64::NAN), Err(FieldError::InvalidTime));
        assert_eq!(layer.evolve(&[1.0], 0.1), Err(FieldError::DimensionMismatch));
        assert_eq!(layer.state, vec![0.0, 0.0]);
    }
}

#[cfg(test)]
mod distance_integrity_tests {
    use super::*;

    #[test]
    fn distance_rejects_nonfinite_intermediate_result() {
        let manifold = PoincareBall::new(1, 1.0).unwrap();
        let x = vec![0.0];
        let y = vec![0.999_999_999_999];
        let distance = manifold.distance(&x, &y).unwrap();
        assert!(distance.is_finite());
    }
}

#[cfg(test)]
mod clock_integrity_tests {
    use super::*;

    #[test]
    fn step_rejects_invalid_existing_clock() {
        let manifold = PoincareBall::new(1, 1.0).unwrap();
        let mut fiber = SystemFiber::new("x", manifold, vec![
            SubsystemLayer::new("layer", vec![0.0]),
        ]).unwrap();
        fiber.engine_clock = f64::NAN;
        assert_eq!(fiber.step(0.1), Err(FieldError::InvalidTime));
    }

    #[test]
    fn evolve_rejects_negative_existing_clock() {
        let manifold = PoincareBall::new(1, 1.0).unwrap();
        let mut fiber = SystemFiber::new("x", manifold, vec![
            SubsystemLayer::new("layer", vec![0.0]),
        ]).unwrap();
        fiber.engine_clock = -1.0;
        assert_eq!(fiber.evolve(&[vec![0.0]], 0.1), Err(FieldError::InvalidTime));
    }
}

#[cfg(test)]
mod matrix_distance_regression {
    use super::*;

    #[test]
    fn matrix_multiply_handles_cancelling_extreme_products() {
        let a = vec![vec![1.0e308, 1.0e308], vec![0.0, 0.0]];
        let b = vec![vec![1.0e308, 0.0], vec![-1.0e308, 0.0]];
        let product = mat_mul(&a, &b).unwrap();
        assert_eq!(product, vec![vec![0.0, 0.0], vec![0.0, 0.0]]);
    }

    #[test]
    fn matrix_distance_handles_extreme_finite_differences() {
        let a = vec![vec![5.0e307, -5.0e307]];
        let b = vec![vec![-5.0e307, 5.0e307]];
        let distance = matrix_distance(&a, &b).unwrap();
        assert!(distance.is_finite());
        assert!(distance > 0.0);
    }

    #[test]
    fn matrix_distance_rejects_non_finite_difference() {
        let a = vec![vec![f64::MAX]];
        let b = vec![vec![-f64::MAX]];
        assert_eq!(matrix_distance(&a, &b), Err(FieldError::NonFiniteState));
    }
}

fn matrix_inverse(a:&Matrix)->Result<Matrix,FieldError>{let n=a.len();if n==0||a.iter().any(|r|r.len()!=n||r.iter().any(|v|!v.is_finite())){return Err(FieldError::InvalidTransport)}let scale=a.iter().flatten().fold(0.0_f64,|m,&v|m.max(v.abs()));if scale==0.0{return Err(FieldError::NonInvertibleTransport)}let pivot_tol=EPS*scale;let mut aug=vec![vec![0.0;2*n];n];for i in 0..n{for j in 0..n{aug[i][j]=a[i][j];}aug[i][n+i]=1.0;}for col in 0..n{let mut pivot=col;for row in col+1..n{if aug[row][col].abs()>aug[pivot][col].abs(){pivot=row;}}if aug[pivot][col].abs()<=pivot_tol{return Err(FieldError::NonInvertibleTransport)}aug.swap(col,pivot);let p=aug[col][col];for j in 0..2*n{aug[col][j]/=p;}for row in 0..n{if row==col{continue}let factor=aug[row][col];for j in 0..2*n{aug[row][j]-=factor*aug[col][j];}}}if aug.iter().flatten().any(|v|!v.is_finite()){return Err(FieldError::NonFiniteState)}Ok(aug.into_iter().map(|row|row[n..].to_vec()).collect())}

#[derive(Debug,Clone,PartialEq)]
pub enum FieldError{DimensionMismatch,GeometryMismatch,InvalidGeometry,InvalidPoint,InvalidTransport,TransportMismatch,MissingTransport,NonInvertibleTransport,NonFiniteState,LayerMismatch,EmptyState,InvalidTime,InvalidDynamics,DuplicateId}
impl fmt::Display for FieldError{fn fmt(&self,f:&mut fmt::Formatter<'_>)->fmt::Result{write!(f,"{self:?}")}}
impl std::error::Error for FieldError{}

#[cfg(test)]
mod tests {
    #[test]
    fn failed_step_does_not_partially_commit_layer_state_or_clock() {
        let manifold = PoincareBall::new(1, 1.0).unwrap();
        let mut fiber = SystemFiber::new(
            "x",
            manifold,
            vec![SubsystemLayer::new("a", vec![0.25])],
        ).unwrap();
        fiber.engine.biases[0][0] = f64::MAX;
        let before = fiber.clone();
        assert_eq!(fiber.step(1.0), Err(FieldError::NonFiniteState));
        assert_eq!(fiber, before);
    }


    use super::*;

    #[test]
    fn distance_invariants_hold() {
        let m = PoincareBall::new(3, 1.0).unwrap();
        let x = vec![0.1, -0.2, 0.05];
        let y = vec![-0.15, 0.1, 0.2];
        assert!(m.distance(&x, &x).unwrap().abs() < 1.0e-12);
        assert!((m.distance(&x, &y).unwrap() - m.distance(&y, &x).unwrap()).abs() < 1.0e-12);
        assert!(m.distance(&x, &y).unwrap() > 0.0);
    }

    #[test]
    fn distance_is_consistent_with_origin_log() {
        let m = PoincareBall::new(2, 1.0).unwrap();
        let x = vec![0.2, -0.3];
        let d = m.distance(&[0.0, 0.0], &x).unwrap();
        let tangent = m.log_origin(&x).unwrap();
        assert!((d - norm(&tangent)).abs() < 1.0e-12);
    }

    #[test]
    fn exp_and_log_reject_nonfinite_inputs_with_state_error() {
        let m = PoincareBall::new(2, 1.0).unwrap();
        assert_eq!(m.exp_origin(&[f64::NAN, 0.0]), Err(FieldError::NonFiniteState));
        assert_eq!(m.log_origin(&[f64::NAN, 0.0]), Err(FieldError::InvalidPoint));
    }

    #[test]
    fn exp_log_round_trip() {
        let m = PoincareBall::new(2, 1.0).unwrap();
        let x = vec![0.2, -0.3];
        let tangent = m.log_origin(&x).unwrap();
        let recovered = m.exp_origin(&tangent).unwrap();
        for (a, b) in x.iter().zip(recovered.iter()) { assert!((a-b).abs() < 1.0e-12); }
    }

    #[test]
    fn exp_log_round_trip_near_boundary() {
        let m = PoincareBall::new(2, 1.0).unwrap();
        let x = vec![0.999_999, 0.0];
        let tangent = m.log_origin(&x).unwrap();
        let recovered = m.exp_origin(&tangent).unwrap();
        assert!((x[0]-recovered[0]).abs() < 1.0e-9);
        assert!(m.contains(&recovered));
    }

    #[test]
    fn project_rejects_finite_coordinates_whose_norm_overflows() {
        let m = PoincareBall::new(4, 1.0).unwrap();
        let x = vec![f64::MAX; 4];
        assert_eq!(m.project(&x), Err(FieldError::NonFiniteState));
    }

    #[test]
    fn geometry_norm_handles_large_finite_coordinates() {
        let m = PoincareBall::new(2, 1.0e-300).unwrap();
        let p = vec![1.0e149, 1.0e149];
        assert!(m.contains(&p));
        let v = m.log_origin(&p).unwrap();
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn distance_handles_large_coordinates_without_overflow() {
        let m = PoincareBall::new(2, 1.0e-300).unwrap();
        let x = vec![1.0e149, 0.0];
        let y = vec![-1.0e149, 0.0];
        let d = m.distance(&x, &y).unwrap();
        assert!(d.is_finite());
        assert!(d > 0.0);
    }

    #[test]
    fn exp_log_round_trip_extreme_tangent_is_finite() {
        let m = PoincareBall::new(2, 1.0).unwrap();
        let v = vec![1.0e308, 0.0];
        let x = m.exp_origin(&v).unwrap();
        assert!(x.iter().all(|value| value.is_finite()));
        assert!(m.contains(&x));
    }

    #[test]
    fn barycenter_handles_cancelling_extreme_tangents() {
        let m = PoincareBall::new(2, 1.0e-300).unwrap();
        let a = vec![1.0e149, 0.0];
        let b = vec![-1.0e149, 0.0];
        let center = m.origin_barycenter(&[a, b]).unwrap();
        assert!(center.iter().all(|v| v.is_finite()));
        assert!(norm(&center) < 1.0e-12);
    }

    #[test]
    fn empty_fiber_has_no_aggregate_system_state() {
        let m = PoincareBall::new(1, 1.0).unwrap();
        let fiber = SystemFiber::new("empty", m, Vec::new()).unwrap();
        assert_eq!(fiber.state(), Err(FieldError::EmptyState));
    }

    #[test]
    fn inverse_accepts_uniformly_scaled_invertible_transport() {
        let matrix = vec![vec![1.0e-15, 0.0], vec![0.0, 2.0e-15]];
        let inverse = matrix_inverse(&matrix).unwrap();
        assert!((inverse[0][0] - 1.0e15).abs() / 1.0e15 < 1.0e-12);
        assert!((inverse[1][1] - 5.0e14).abs() / 5.0e14 < 1.0e-12);
    }

    #[test]
    fn inverse_rejects_singular_transport() {
        let matrix = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        assert_eq!(matrix_inverse(&matrix), Err(FieldError::NonInvertibleTransport));
    }

    #[test]
    fn fiber_step_rejects_mutated_invalid_engine() {
        let m = PoincareBall::new(1, 1.0).unwrap();
        let mut fiber = SystemFiber::new("x", m, vec![SubsystemLayer::new("a", vec![0.0])]).unwrap();
        fiber.engine.damping = vec![-1.0];
        assert_eq!(fiber.step(0.1), Err(FieldError::InvalidDynamics));
    }

    #[test]
    fn fiber_rejects_nonfinite_layer_state() {
        let m = PoincareBall::new(1, 1.0).unwrap();
        let layer = SubsystemLayer { id: "a".into(), state: vec![f64::NAN], gain: 1.0 };
        assert_eq!(SystemFiber::new("x", m, vec![layer]), Err(FieldError::NonFiniteState));
    }

    #[test]
    fn fiber_rejects_duplicate_layer_ids() {
        let m = PoincareBall::new(1, 1.0).unwrap();
        let layers = vec![
            SubsystemLayer::new("a", vec![0.0]),
            SubsystemLayer::new("a", vec![0.0]),
        ];
        assert_eq!(SystemFiber::new("x", m, layers), Err(FieldError::DuplicateId));
    }

    #[test]
    fn fiber_rejects_nonfinite_layer_gain() {
        let m = PoincareBall::new(1, 1.0).unwrap();
        let layer = SubsystemLayer { id: "a".into(), state: vec![0.0], gain: f64::NAN };
        assert_eq!(SystemFiber::new("x", m, vec![layer]), Err(FieldError::InvalidDynamics));
    }
}

#[cfg(test)]
mod global_step_tests {
    use super::*;

    fn identity_transport(from: &str, to: &str) -> GaugeTransport {
        GaugeTransport::new(from, to, vec![vec![1.0]]).unwrap()
    }

    #[test]
    fn global_step_evolves_fibers_then_synchronizes() {
        let manifold = PoincareBall::new(1, 1.0).unwrap();
        let first = SystemFiber::new(
            "first",
            manifold.clone(),
            vec![SubsystemLayer::new("state", vec![0.1])],
        ).unwrap();
        let second = SystemFiber::new(
            "second",
            manifold.clone(),
            vec![SubsystemLayer::new("state", vec![0.2])],
        ).unwrap();

        let mut field = GlobalField::new(manifold);
        field.integration_gain = 1.0;
        field.cooling_rate = 0.0;
        field.add_fiber(first).unwrap();
        field.add_fiber(second).unwrap();
        field.add_transport(identity_transport("second", "first")).unwrap();

        let consensus = field.step(0.1).unwrap().unwrap();

        assert!((field.fibers[0].engine_clock - 0.1).abs() < 1.0e-12);
        assert!((field.fibers[1].engine_clock - 0.1).abs() < 1.0e-12);
        assert!(consensus[0] > 0.1 && consensus[0] < 0.2);
        assert!(field.fibers[1].state().unwrap()[0] < 0.2);
        assert!(field.disagreement.is_finite());
    }

    #[test]
    fn global_step_is_transactional_across_fibers() {
        let manifold = PoincareBall::new(1, 1.0).unwrap();
        let first = SystemFiber::new(
            "first",
            manifold.clone(),
            vec![SubsystemLayer::new("state", vec![0.1])],
        ).unwrap();
        let second = SystemFiber::new(
            "second",
            manifold.clone(),
            vec![SubsystemLayer::new("state", vec![0.2])],
        ).unwrap();

        let mut field = GlobalField::new(manifold);
        field.add_fiber(first).unwrap();
        field.add_fiber(second).unwrap();
        field.add_transport(identity_transport("second", "first")).unwrap();
        field.fibers[1].engine.damping[0] = f64::NAN;

        let before_first_state = field.fibers[0].state().unwrap();
        let before_second_state = field.fibers[1].state().unwrap();
        let before_first_clock = field.fibers[0].engine_clock;
        let before_second_clock = field.fibers[1].engine_clock;
        let before_disagreement = field.disagreement;
        let before_temperature = field.temperature;
        let before_cooling_rate = field.cooling_rate;

        assert_eq!(field.step(0.1), Err(FieldError::InvalidDynamics));

        assert_eq!(field.fibers[0].state().unwrap(), before_first_state);
        assert_eq!(field.fibers[1].state().unwrap(), before_second_state);
        assert_eq!(field.fibers[0].engine_clock, before_first_clock);
        assert_eq!(field.fibers[1].engine_clock, before_second_clock);
        assert!(field.fibers[1].engine.damping[0].is_nan());
        assert_eq!(field.disagreement, before_disagreement);
        assert_eq!(field.temperature, before_temperature);
        assert_eq!(field.cooling_rate, before_cooling_rate);
    }
}
