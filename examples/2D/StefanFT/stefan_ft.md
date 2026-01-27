# Lists of files

- examples/2D/StefanFT/stefan_deca.jl : Circular front-tracking Stefan problem of a decaying solid phase : Ok
- examples/2D/StefanFT/crystal_growth.jl : Circular front-tracking Stefan problem of a growing solid phase : Ok
- examples/2D/StefanFT/crystal_decay.jl : Circular front-tracking Stefan problem of a decaying solid phase : Ok (but add a front tracking routine to remove markers that are too close)

## stefan_deca.jl

- NX, NY : 32,32 
- Newton max iter : 1
- N markers : 100
=> Developpement of mullins sekerka instability
- Break due to markers too close and overlap => Need to reinject markers where curvature is high


## crystal_growth.jl



---
- BE : trop diffusif vs algo mise a jour de l'interface
- connectivté marker fin remettre au debut 
- associer marker une cellule et une level set par cellule
- cell merging : 45 / 90