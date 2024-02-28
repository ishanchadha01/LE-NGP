init_trainer
    setup_model
    setup_optim
    setup_sched
    wrap model in distributed trainer

setup_model
    gets model
    creates model

gets model
    cfg, initializes weights, move it to CUDA, *affected by cfg! need to adjust
    ints loss funcs - L1 Loss fo rrender key in criteria dict
    other bookkeepig like checkpointer

creates model
    model init for neuralangelo
        some cfg stuff
        build model
            adds appearance enc if enabled
            neural rgb init
            neural sdf init
            background nerf is same as RGB, *maybe turn it off!

neural sdf init
    enc dim = build enc, makes tcnn hash enc
    inp dim = 3 + enc dim
        3 is for points3d, is actually \[b,r,n,3\]
        enc is samples, \[b,r,n,ld\]
    mlp = mlp4neuralsdf, this builds hash enc and tcnn stuff unlike neuralrgb

mlp4neuralsdf
    output is 1 bc just sdf, so \[b,r,n,1\], right?
    inits weights a certain way called geometric init, prolly mathematically better for convergence or something
    also supports skips, just concats to layer when skip connects
    *building these should be easy! very configurable, i also think hash grids are just features, and smoothing how backprop is distributed along the whole net is what intermediate splatting can be done (FAST!)

neural rgb init
    torch nn module
    enc view dim = build encoding
        build encoding just makes data structure which is filled in, very efficient!
            build enc uses cfg to either create fourier transform or spherical harmonics
        inp dim = inp base dim + enc view dim + feat dim (+ appear dim)
            inp base dim is 3, or 6 if implicit differentiable renderer (IDR) turned on
            feat dim is sdf MLP hidden dim
        build MLP
            layers dims: inputs->hidden_dim*num_layers->3(color)
            select activation
            MLP w Skip
                its just a regular MLP! only SDF uses tcnn when doing hash encoding





set train dataloader
    get train dataset objects
    create neuralangelo dataset, loads info abt imgs like num_frames, HW, num_rays, provides methods to preprocess img, get cam center, get c2w transform
    created distributed wrapper around dataloader

get train dataset objects
    create cerf dataset, just torch dataset with multithreaded data preloader, does it even work cuz of GIL?

set val dataloader, same as set train dataloader

trainer checkpointer load, generally checkpointer stores info at training and was initialized earlier, *can be used to load model weights if i do ROS thing!

trainer.train
    nerf trainer train
        get data abt curr epoch from checkpointer
        if val epoch, self.test, and log via wandb
    call imaginaire trainer.train

imaginaire trainer.train
    for epoch in epochs, for batch in dataset, train_step wrapped in start_of_iter and end_of_iter which is just for logging p much

train_step
    model_forward(data)
        self.model(data) from nerf trainer
            return model.render_pixels->ray conversion
                center, ray = \[B, HW, 3\],\[B, HW, 3\]  *print to ensure ray is per pixel and centers are just replicated into that shape
                what is ray idx, ray, and ray unit? so basically, dataset from data.py does this thing where, when requested a sample. it gets a torch.randperm using H and W to get ray_idx of size \[R\], along with a sampled img and its corresponding pose, and output dict, referenced as "data" since thats how its named after being retrieved from dataloader. slightly different for inference, this is for train. one more thing is that image is returned as samples along these rays, so \[R,3\] as opposed to HW, also note that although its created as r,3, its retrieved as b,r,3
                slice_by_ray_idx to get centers[b_idx, r_idx] from the sample retrieved and rays[b_idx, r_idx], both of size [b,r,3 ]. these are called center and ray,theyre one img center (replicated) and rays for that img *print to check this
                ray unit is when when ray is normalized

                * if i didnt sample along rays, i could just have a pixel-wise sampling, which could serve as sparse point cloud

                render rays with center, ray unit, sample idx(just idx of sample in dataset), stratified *what is stratified

render rays
    near, far, outside gotten using get dist bounds
    appearnace mebedding gotten if required
    render rays obj, gets sdfs, rgbs
    weights = alpha compositing weights like in render ray obj for sampling
    rgb sis composite rendering using rgb output from render ray obj and weights, * this is prolly where alpha blending would be used


render rays obj
    sample dists all
        dists gotten of samples by passing in center, ray unit, near and far. which distance? dist is dist along its respective ray, sampled in range near,far. size b,r,n,1. 
        if num sample hierarchy is >0, then we get sdfs as well as sdfs fine. using get 3d points from dist, we get the 3d points in space of these dists *exactly what the sparse point cloud is!!!
        now we have points and sdfs for original set of dists, now call sample dists hierarchical using initial sdf and points to get sdf fine and point fine, concatting together to get hierarchical coarse to fine dists output, and sdfs, *the coarse to find sdfs can prolly all be passed in, is good way of sampling nonfree space more i believe
        sample dists hierarchical uses alpha compositing weights to weight dists by cdf, and then samples from pdf

        uses dists to compute 3d points, and use points to get sdfs and feats using neural sdf forward pass. 
        compute gradients, hessiants
        normals are just gradients

        rgbs = neural rgb forward pass *this part needs to be replaced with splatting
            pass in points, normals, rays unit, feats, and appearance if used
        
        sdf volume rendering with compute neus alphas



                render rays, testing constructing loss