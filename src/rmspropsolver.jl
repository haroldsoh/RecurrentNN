
type RMSPropSolver <: Solver
   decayrate::Float64
   smootheps::Float64
   stepcache::Array{NNMatrix,1}
   RMSPropSolver() = new(0.999, 1e-8, Array(NNMatrix,0))
end

type RMSPropSolverParams <: SolverParams
    regc::Float64
    learning_rate::Float64
    clipval::Float64
end

function step!(solver::RMSPropSolver, model::Model, sparams::RMSPropSolverParams)

    # perform parameter update
    solverstats = Array(Float64,0)
    numclipped = 0
    numtot = 0

    # All of the matrices used by the model
    modelMatices = model.matrices

    # init stepcache if needed
    if length(solver.stepcache) == 0
         for m in modelMatices
            push!(solver.stepcache, NNMatrix(m.n, m.d))
        end
    end

    for k = 1:length(modelMatices)
        @inbounds m = modelMatices[k] # mat ref
        @inbounds s = solver.stepcache[k]
        for i = 1:m.n
            for j = 1:m.d

                # rmsprop adaptive learning rate
                @inbounds mdwi = m.dw[i,j]
                @inbounds s.w[i,j] = s.w[i,j] * solver.decayrate + (1.0 - solver.decayrate) * mdwi^2

                # gradient clip
                if mdwi > sparams.clipval
                    mdwi = sparams.clipval
                    numclipped += 1
                end

                if mdwi < -sparams.clipval
                    mdwi = -sparams.clipval
                    numclipped += 1
                end
                numtot += 1

                # update (and regularize)
                @inbounds m.w[i,j] += - sparams.learning_rate * mdwi / sqrt(s.w[i,j] + solver.smootheps) - sparams.regc * m.w[i,j]
                @inbounds m.dw[i,j] = 0. # reset gradients for next iteration
            end
        end
    end
    solverstats　=  numclipped * 1.0 / numtot
    return solverstats
end
