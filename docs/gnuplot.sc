f(x,y) = (y > 0 ? ((-4 -2*x + y > 0)?1:1/0) : 1/0) + (y>0?((4 + 2*x + y) > 0 ? 1: 1/0 ): 1/0) + (y>0?((4 - x - 2*y)>0? 1: 1/0):1/0) + (y>0?((9 + 2*x - y)>0 ? 1: 1/0): 1/0) 
set terminal pdf
unset colorbox
set isosample 500, 500
set sample 500
set pm3d map
set output "relu_nonl.pdf"
splot [-5:5] [-5:5] f(x,y)
