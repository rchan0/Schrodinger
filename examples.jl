# might have to use include to import module
using .MySchrodinger
using Plots

# solve Schrodinger for square well with default values
psi, evals, unbound = solve(x-> squarewell(x))

# time evolve superposition of 1st, 2nd, 3rd states
trange1=range(0,step=0.005, stop=2)
anim1 = @animate for t in trange1
    plot(x, prob((timeevolution(psi[1],evals[1], t)+timeevolution(psi[2],evals[2], t)+timeevolution(psi[3],evals[3], t))./sqrt(3), dx))
    ylims!((0,0.004))
end
g1=gif(anim1, fps = 8);

# create fake arbitrary wavefunction
ps=(psi[1]+psi[2]+psi[3]+psi[4])./2

# decompose into stationary states
x=range(-1, step=dx, stop=1);
coeffs=decompose(ps,y->squarewell(y),x)

# time evolve ps
trange2=range(0,step=0.01, stop=2)
anim2 = @animate for t in trange2
    plot(x, prob(sum([coeffs[i].*timeevolution(psi[i],evals[i],t) for i=1:length(coeffs)]), dx))
    ylims!((0,0.0045))
end
g2=gif(anim2, fps = 8);


