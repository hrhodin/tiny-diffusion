# Experiments

1. 2D sphere single view: observed 2D projection (initially we experimented with a constant unknown observation, 
  only backpropagating through the loss) 
   * A single projection does not constrain the solution uniquely. The ideal model would predict all possible results
     with an equal probability (to be tested, initial tests at least improve)
2. 2D sphere stereo: conditioned on a second observation (at training time only?), with unknown camera pose
  (making camera pose a latent variable and also conditioning on the rendering?)?
3. For complex scenes consisting of multiple objects, would it be possible to learn 3D from unrelated monocular 2D views?
  - this required in the past to use a canonical frame and to enforce similarity between instances
    (Possibilities: sparse differences? Lowest entropy across batch? ...)
  - could require multiple sampling and only keeping the best direction?
4. Does this help beyond inverse graphics problems?
   - the decoder above could be learned too? Would it collapse in the unknown view case? - Yes it would essentially become 
     an image model again? Or not, if we enforce that the latent space is structured (small, hierarchical... ?)
   - could we learn latent diffusion end-to-end? We would still need an encoder to analyze the image 
     (as in the first examples above with predefined decoder=renderer). But now it would make sense to learn the 
     decoder alongside, and also the encoder could be learned. Possibly leading to a rich feature extractor. Open question
     of whether more capacity should be put on the encoder or the diffusion model modeling distributions within the latent
     space.
   - would that be interesting for language? One could possibly learn a more meaningful latent space and try to make it
     interpretable?
5. Why could the whole thing fly/work?
   - The noise natural to diffusion models leads to an exploration of previously unknown spaces, helping to 
     solve difficult inverse problems.
   - The probabilistic nature enables learning uncertainty explicitly and hence solve problems with multiple, 
     infinnitely many solutions by sampling from the possible subset.
   - The iterative nature breaks down a complex solution path into smaller parts
6. How could it fail, what are difficulties?
   - The diffusion approch requires an estimate of x_0, to add noise to. It is expensive to compute this and is only
     (approximatively) available once the model is trained - a chicken and egg problem?
   - We do not train the model to predict the true x_0 (or corresponding noise) but only to predict the best possible
     within the noise radius (is equivalent in theory only if there is a single solution that it best, in the case of 
     uncertainty, we require the local search to prevent collapse)
   - Its not clear how to constrain the prediction to follow the desired eps distribution (normalization, bounding, ...)
     nor what the effect of doing so is on the solution. What we can say for sure is that it spreads uncertainty, but 
     how much? How much remain predictions biased?
   - Already in the most simple case we have seen that some solution areas are ignored by the model if the initializaiton
     already moves points away from it. It may require training an assemble of models or to actively seek for coverage?

Differences to Ayush's method?
1. Not needing an encoder from image space, diffusion happens in latent space - more efficient? 
  * Does it still need an image encoder to assess the alignment? It could be a frozen DINO etc.
2. May need a feedback to the denoiser of how good the current latent space projects on the synthetic view
   * In that case, are we still doing normal diffusion or are we just learning to learn (learning to optimize)
   * I don't think the model would need this explicit rendering feedback loop. The model could internally reason how the 
     predicted latent code would look for the predicted camera and then reason about the difference to predict the next.
   * Hence, this explicit feedback could help practically but does not alter the theoretical framework, the model has
     already all the relevant information!
 - So we would have the speed and simplicity advantage. Are we modeling the same distribution?
  * I would argue that they are not properly doing a diffusion process since noise is added in the image domain, not shape/latent domain
    Its at best a projection of the latent distribution. Each time, the encoder has to read the current image stage,
    predict what the latent is (which is ambigious by definition) and then refine. One could potentially do it on multiple 
    views to make it well-defined. 
  * If views are predicted alongside, it would again be hard for them to model as only very indirectly defined in the renderings
 - So showing that we can estimate unobserved camera angles will be key to showing the advantages of this model

Side notes:
 - OpenBrain prediction of future. Brought about the idea that machines would reason using natural language to be interpretable? 
 - It is certainly an interesting thought and step in the right direction. It could still encode hidden information in text. 
 - It could simply explain decisions with some sensible text but the real reason might be an other one (as humans do a lot)
  - But perhaps that could be analyzed somehow by looking at the NN activations (if in some structured form?!) 