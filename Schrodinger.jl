module MySchrodinger
using LinearAlgebra

export solve, timeevolution, decompose, momentum, prob, squarewell, asymwell, wellbarrier

function solve(V::Function, x=range(-1, step=0.001, stop=1))
    # V = potential function
    # x = range of x values to solve equation on
    # returns a tuple of an array of position-space wavefunction values, 
    # their corresponding eigenvalues, and a number indicating the first unbound state, if any
    # uses hbar=mass=1
    n=length(x)
    dx=x[2]-x[1]
    
    # discretize Schrodinger equation
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
    
    # solve for energy eigenvalues and wavefunctions
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
    # helper function that normalizes input
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
    # t = time value
    return psi.*exp(-1im*E*t)
end

function decompose(psi, V, x)
    # decomposes psi into superposition of eigenstates corresponding to potential V
    # returns the coefficients corresponding to this superposition
    # psi = arbitrary wavefunction
    # x = range of x values corresponding to psi
    vecs=solve(y->V(y),x)[1]
    coeffs=inv(reshape(hcat(vecs...), (length(vecs[1]), length(vecs))))*psi
    return coeffs
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
    # returns probability of psi
    # dx = step for x values
    return abs2.(psi)*dx
end
end
