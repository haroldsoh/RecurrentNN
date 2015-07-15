type AdamSolver <: Solver
  timestep::Int64
  alpha::Float64
  beta1::Float64
  beta2::Float64
  epsilon::Float64

  mts::Array{Any}
  vts::Array{Any}

  AdamSolver() = new(1, 0.001, 0.9, 0.999, 1e-8, [], [])
end

type AdamSolverParams <: SolverParams
    clipval::Float64
end

function step!(solver::AdamSolver, model::Model, sparams::AdamSolverParams)

  # perform parameter update

  # All of the matrices used by the model
  modelMatices = model.matrices

  # init
  if length(solver.mts) == 0
    for m in modelMatices
      push!(solver.mts, zeros(m.n, m.d))
    end
  end

  if length(solver.vts) == 0
    for m in modelMatices
      push!(solver.vts, zeros(m.n, m.d))
    end
  end
  alpha = solver.alpha;
  beta1 = solver.beta1;
  beta2 = solver.beta2;
  at = alpha*(sqrt(1-(beta2^solver.timestep)))/(1 - beta1^solver.timestep);

  for k = 1:length(modelMatices)
    @inbounds A = modelMatices[k] # mat ref
    @inbounds mt = solver.mts[k]
    @inbounds vt = solver.vts[k]
    for i = 1:A.n
      for j = 1:A.d

        # get the gradients
        @inbounds gt = A.dw[i,j]

        # gradient clip
        if gt > sparams.clipval || gt < -sparams.clipval
          gt = sparams.clipval*sign(gt);
        end

        # update biased 1st moment estimate
        @inbounds mt[i,j] = beta1*mt[i,j] + (1-beta1)*gt;
        @inbounds vt[i,j] = beta2*vt[i,j] + (1-beta2)*(gt.*gt);

        # update
        @inbounds A.w[i,j] += -at * mt[i,j]/(sqrt(vt[i,j]) + solver.epsilon);
        @inbounds A.dw[i,j] = 0. # reset gradients for next iteration

      end
    end
  end
  solver.timestep = solver.timestep + 1;
  return
end
