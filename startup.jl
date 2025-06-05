begin
	using Pkg
	Pkg.activate(".")
	Pkg.instantiate	
	using SchurParlett
	using Revise
	includet("src/SchurParlett.jl")
end
