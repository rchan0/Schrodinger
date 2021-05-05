module MySchrodinger
using LinearAlgebra

export solve, timeevolution, decompose, timeevol_psi, momentum, prob, squarewell, asymwell, wellbarrier

function solve(V::Function, x=range(-1, step=0.001, stop=1))
    # V = potential function
    # x = range of x values to solve equation on
    # returns a tuple of an array of position-space wavefunction values, 
    # their corresponding eigenvalues, and a number indicating the first unbound state, if any
    # uses hbar=mass=1
    n=length(x)
    dx=x[2]-x[1]
    m=zeros(n,n)
    k=1/(2*dx^2)
    @inbounds for i=1:n
        m[i,i]=2*k+V(x[i])
        if i==1
            m[i,i+1]=-k
        elseif i==n
            m[i,i-1]=-k
        else 
            m[i,i+1]=-k
            m[i,i-1]=-k
        end
    end
    evals,evecs=eigen(m)

    # only normalize bound states
    vmax=maximum(V.(x))
    unbound=-1
    @inbounds for i=1:n
        if evals[i]>vmax
            unbound=i
            break
        end
    end
    psis=[normalizepsi(evecs[1:n,i],dx) for i=1:unbound-1]
    # add the unbound states to the array 
    return (vcat(psis,[evecs[1:n,i] for i=unbound:n]), evals, unbound)
end

function normalizepsi(psi,dx)
    area=sum(abs2.(psi))*dx
    psi=psi./sqrt(area)
    return psi
end

function squarewell(x; v0=100, L=1)
    # finite square well centered at x=0
    # v0 = potential outside well; default is 100
    # L = length of box; default is 1
    if real(v0)!=v0|| real(L)!=L || L<=0
        throw(DomainError()) 
    end
    if abs(x)>L/2
        return v0
    else
        return 0
    end
end

function asymwell(x; v0=100, v2=50, pos=0, L=1)
    # asymmetric well centered at x=0
    # v0 = potential outside well; default is 100
    # v2 = potential inside part of well; default is 50
    # pos = position indicating asymmetry
    if real(v0)!=v0 || real(L)!=L || L<=0 || abs(pos)>=L/2
        throw(DomainError()) 
    end
    if abs(x)>L/2
        return v0
    elseif x<pos
        return v2
    else 
        return 0
    end
end

function wellbarrier(x;v0=100,vb=70,L=1,w=0.4)
    # w = width of barrier, centered at x=0
    # vb = height of barrier
    if real(v0)!=v0 || real(vb)!=vb ||real(L)!=L || L<=0 || w<0 || w>L
        throw(DomainError()) 
    end
    if abs(x)<w/2
        return vb
    elseif abs(x)>L/2
        return v0
    else
        return 0
    end
end

function timeevolution(psi, E, t)
    # time evolution of a stationary state
    # E = energy eigenvalue corresponding to psi
    return psi.*exp(-1im*E*t)
end

function decompose(psi, V, xrange)
    # psi = arbitrary wavefunction corresponding to potential function V
    # xrange = range of x values corresponding to psi
    # decomposes psi into linear combination of eigenstates
    # returns the coefficients corresponding to this superposition
    vecs=solve(x->V(x),xrange)[1]
    coeffs=inv(reshape(hcat(vecs...), (length(vecs[1]), length(vecs))))*psi
    return coeffs
end

function timeevol_psi(psi, t, V, xrange)
    # time evolution of an arbitrary wavefunction
    evecs, evals=solve(x->V(x),xrange)
    coeffs=decompose(psi,x->V(x),xrange)
    return sum([coeffs[i].*timeevolution(evecs[i],evals[i],t) for i=1:length(coeffs)])
end

function momentum(psi; x=range(-1, step=0.001, stop=1), p=range(-30, step=0.05, stop=30))
    # psi = 1D array of position wavefunction values
    # x = range of x values that correspond to psi
    # p = range of momentum values
    n=length(x)
    dx=x[2]-x[1]
    momentum=[]
    for pn in p
        push!(momentum,sum([dx*psi[i]*exp(-pn*x[i]*1im) for i=1:n])/sqrt(2*pi))
    end
    return momentum
end

function prob(psi, dx)
    return abs2.(psi)*dx
end
end

using .MySchrodinger

# import Pkg; Pkg.build("GR")
using Plots

(psi, evals, unbound)=solve(x-> squarewell(x));

unbound

dx=0.001
x=range(-1, step=dx, stop=1)
plot(x,psi[1], label=1)
plot!(x,psi[2], label=2)
plot!(x,psi[5], label=5)
plot!(x,psi[6], label=6)

dp=0.05
p=range(-30, step=dp, stop=30)
plot(p,prob(momentum(psi[1]),dp), label=1)
plot!(p,prob(momentum(psi[2]),dp), label=2)
plot!(p,prob(momentum(psi[3]),dp), label=3)

sum(prob(momentum(psi[2]),dp))

trange=range(0,step=0.005, stop=2)
anim = @animate for t in trange
    plot(x, prob((timeevolution(psi[1],evals[1], t)+timeevolution(psi[2],evals[2], t)+timeevolution(psi[3],evals[3], t))./sqrt(3), dx))
    ylims!((0,0.004))
end
g1=gif(anim, fps = 8);

# g1

anim = @animate for t in trange
    plot(x, prob(timeevolution(psi[2],evals[2], t)./sqrt(3/2)+timeevolution(psi[5],evals[5], t)./sqrt(3), dx))
    ylims!((0,0.004))
end
g2=gif(anim, fps = 8);

# g2

(psi1, evals1, unbound1)=solve(x-> squarewell(x, v0=10000000));

plot(x,psi1[1], label=1)
plot!(x,psi1[2], label=2)

evals1[1:10]

sqrt.(evals1[1:10]./evals1[1])

(psi2, evals2, unbound2)=solve(x-> wellbarrier(x));

plot(x,[wellbarrier(i) for i in x])

plot(x,psi2[1],label=1)
plot!(x,psi2[2],label=2)
plot!(x,psi2[3],label=3)

trange=range(0,step=0.01, stop=10)
anim = @animate for t in trange
    plot(x, prob((timeevolution(psi[1],evals[1], t)+timeevolution(psi[2],evals[2], t)+timeevolution(psi[3],evals[3], t))./sqrt(3), dx))
    ylims!((0,0.0045))
end

g3=gif(anim, fps = 8);




