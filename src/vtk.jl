"""
    write_vtk(filename::String, mesh::AbstractMesh, solver::Solver)

Writes a VTK file based on the solver's time, phase, and equation types.

# Arguments
- `filename::String` : Name of the VTK file to create.
- `mesh::AbstractMesh` : Mesh used in the simulation.
- `solver::Solver` : Simulation results containing the data to write.
"""
function write_vtk(filename::String, mesh::AbstractMesh, solver::Solver)
    if length(mesh.centers) == 1
        if solver.time_type == Steady && solver.phase_type == Monophasic
            # Steady, Monophasic, Diffusion
            vtk_grid(filename, 0:1:length(mesh.centers[1])) do vtk
                vtk["Temperature_b"] = solver.x[1:length(solver.x) ÷ 2]
                vtk["Temperature_g"] = solver.x[length(solver.x) ÷ 2 + 1:end]
                println("VTK file written : $filename.vti")
            end
        elseif solver.time_type == Steady && solver.phase_type == Diphasic
            # Steady, Diphasic, Diffusion
            part = div(length(solver.x), 4)
            vtk_grid(filename, 0:1:length(mesh.centers[1])) do vtk
                vtk["Temperature_1_b"] = solver.x[1:part]
                vtk["Temperature_1_g"] = solver.x[part + 1:2 * part]
                vtk["Temperature_2_b"] = solver.x[2 * part + 1:3 * part]
                vtk["Temperature_2_g"] = solver.x[3 * part + 1:end]
                println("VTK file written : $filename.vti")
            end
        elseif solver.time_type == Unsteady && solver.phase_type == Monophasic
            # Unsteady, Monophasic, Diffusion
            pvd = paraview_collection(filename)
            for (i, state) in enumerate(solver.states)
                vtk_grid(filename * "_$i", 0:1:length(mesh.centers[1])) do vtk
                    vtk["Temperature_b"] = state[1:length(state) ÷ 2]
                    vtk["Temperature_g"] = state[length(state) ÷ 2 + 1:end]
                    pvd[i] = vtk
                end
            end
            vtk_save(pvd)
            println("VTK file written : $filename.pvd")
        elseif solver.time_type == Unsteady && solver.phase_type == Diphasic
            # Unsteady, Diphasic, Diffusion
            pvd = paraview_collection(filename)
            part = div(length(solver.x), 4)
            for (i, state) in enumerate(solver.states)
                vtk_grid(filename * "_$i", 0:1:length(mesh.centers[1])) do vtk
                    vtk["Temperature_1_b"] = state[1:part]
                    vtk["Temperature_1_g"] = state[part + 1:2 * part]
                    vtk["Temperature_2_b"] = state[2 * part + 1:3 * part]
                    vtk["Temperature_2_g"] = state[3 * part + 1:end]
                    pvd[i] = vtk
                end
            end
            vtk_save(pvd)
            println("VTK file written : $filename.pvd")
        else
            error("Combination of TimeType, PhaseType, and EquationType not supported.")
        end
    elseif length(mesh.centers) == 2
        if solver.time_type == Steady && solver.phase_type == Monophasic 
            # Cas Steady, Monophasic, Diffusion
            vtk_grid(filename, 0:1:length(mesh.centers[1]), 0:1:length(mesh.centers[2])) do vtk
                vtk["Temperature_b"] = reshape(solver.x[1:length(solver.x) ÷ 2], length(mesh.centers[1])+1, length(mesh.centers[2])+1)
                vtk["Temperature_g"] = reshape(solver.x[length(solver.x) ÷ 2 + 1:end], length(mesh.centers[1])+1, length(mesh.centers[2])+1)
                println("Fichier VTK steady monophasic écrit : $filename.vti")
            end
        elseif solver.time_type == Steady && solver.phase_type == Diphasic 
            # Cas Steady, Diphasic, Diffusion
            part = div(length(solver.x), 4)
            vtk_grid(filename, 0:1:length(mesh.centers[1]), 0:1:length(mesh.centers[2])) do vtk
                vtk["Temperature_1_b"] = reshape(solver.x[1:part], length(mesh.centers[1])+1, length(mesh.centers[2])+1)
                vtk["Temperature_1_g"] = reshape(solver.x[part + 1:2 * part], length(mesh.centers[1])+1, length(mesh.centers[2])+1)
                vtk["Temperature_2_b"] = reshape(solver.x[2 * part + 1:3 * part], length(mesh.centers[1])+1, length(mesh.centers[2])+1)
                vtk["Temperature_2_g"] = reshape(solver.x[3 * part + 1:end], length(mesh.centers[1])+1, length(mesh.centers[2])+1)
                println("Fichier VTK steady diphasic écrit : $filename.vti")
            end
        elseif solver.time_type == Unsteady && solver.phase_type == Monophasic 
            pvd = paraview_collection(filename)
            # Cas Unsteady, Monophasic, Diffusion
            for (i, state) in enumerate(solver.states)
                vtk_grid(filename * "_$i", 0:1:length(mesh.centers[1]), 0:1:length(mesh.centers[2])) do vtk
                    vtk["Temperature_b"] = reshape(state[1:length(state) ÷ 2], length(mesh.centers[1])+1, length(mesh.centers[2])+1)
                    vtk["Temperature_g"] = reshape(state[length(state) ÷ 2 + 1:end], length(mesh.centers[1])+1, length(mesh.centers[2])+1)
                    pvd[i] = vtk
                end
            end
            vtk_save(pvd)
            println("Fichier VTK unsteady monophasic écrit : $filename.pvd")
        elseif solver.time_type == Unsteady && solver.phase_type == Diphasic 
            pvd = paraview_collection(filename)
            # Cas Unsteady, Diphasic, Diffusion
            part = div(length(solver.x), 4)
            for (i, state) in enumerate(solver.states)
                vtk_grid(filename * "_$i", 0:1:length(mesh.centers[1]), 0:1:length(mesh.centers[2])) do vtk
                    vtk["Temperature_1_b"] = reshape(state[1:part], length(mesh.centers[1])+1, length(mesh.centers[2])+1)
                    vtk["Temperature_1_g"] = reshape(state[part + 1:2 * part], length(mesh.centers[1])+1, length(mesh.centers[2])+1)
                    vtk["Temperature_2_b"] = reshape(state[2 * part + 1:3 * part], length(mesh.centers[1])+1, length(mesh.centers[2])+1)
                    vtk["Temperature_2_g"] = reshape(state[3 * part + 1:end], length(mesh.centers[1])+1, length(mesh.centers[2])+1)
                    pvd[i] = vtk
                end
            end
            vtk_save(pvd)
            println("Fichier VTK unsteady diphasic écrit : $filename.pvd")            
        else
            error("La combinaison de TimeType, PhaseType et EquationType n'est pas supportée.")
        end
    elseif length(mesh.centers) == 3
        if solver.time_type == Steady && solver.phase_type == Monophasic 
            # Cas Steady, Monophasic, Diffusion
            vtk_grid(filename, 0:1:length(mesh.centers[1]), 0:1:length(mesh.centers[2]), 0:1:length(mesh.centers[3])) do vtk
                vtk["Temperature_b"] = reshape(solver.x[1:length(solver.x) ÷ 2], length(mesh.centers[1])+1, length(mesh.centers[2])+1, length(mesh.centers[3])+1)
                vtk["Temperature_g"] = reshape(solver.x[length(solver.x) ÷ 2 + 1:end], length(mesh.centers[1])+1, length(mesh.centers[2])+1, length(mesh.centers[3])+1)
                println("Fichier VTK steady monophasic écrit : $filename.vti")
            end
        elseif solver.time_type == Steady && solver.phase_type == Diphasic 
            # Cas Steady, Diphasic, Diffusion
            part = div(length(solver.x), 4)
            vtk_grid(filename, 0:1:length(mesh.centers[1]), 0:1:length(mesh.centers[2]), 0:1:length(mesh.centers[3])) do vtk
                vtk["Temperature_1_b"] = reshape(solver.x[1:part], length(mesh.centers[1])+1, length(mesh.centers[2])+1, length(mesh.centers[3])+1)
                vtk["Temperature_1_g"] = reshape(solver.x[part + 1:2 * part], length(mesh.centers[1])+1, length(mesh.centers[2])+1, length(mesh.centers[3])+1)
                vtk["Temperature_2_b"] = reshape(solver.x[2 * part + 1:3 * part], length(mesh.centers[1])+1, length(mesh.centers[2])+1, length(mesh.centers[3])+1)
                vtk["Temperature_2_g"] = reshape(solver.x[3 * part + 1:end], length(mesh.centers[1])+1, length(mesh.centers[2])+1, length(mesh.centers[3])+1)
                println("Fichier VTK steady diphasic écrit : $filename.vti")
            end
        elseif solver.time_type == Unsteady && solver.phase_type == Monophasic 
            pvd = paraview_collection(filename)
            # Cas Unsteady, Monophasic, Diffusion
            for (i, state) in enumerate(solver.states)
                vtk_grid(filename * "_$i", 0:1:length(mesh.centers[1]), 0:1:length(mesh.centers[2]), 0:1:length(mesh.centers[3])) do vtk
                    vtk["Temperature_b"] = reshape(state[1:length(state) ÷ 2], length(mesh.centers[1])+1, length(mesh.centers[2])+1, length(mesh.centers[3])+1)
                    vtk["Temperature_g"] = reshape(state[length(state) ÷ 2 + 1:end], length(mesh.centers[1])+1, length(mesh.centers[2])+1, length(mesh.centers[3])+1)
                    pvd[i] = vtk
                end
            end
            vtk_save(pvd)
            println("Fichier VTK unsteady monophasic écrit : $filename.pvd")
        elseif solver.time_type == Unsteady && solver.phase_type == Diphasic 
            pvd = paraview_collection(filename)
            # Cas Unsteady, Diphasic, Diffusion
            part = div(length(solver.x), 4)
            for (i, state) in enumerate(solver.states)
                vtk_grid(filename * "_$i", 0:1:length(mesh.centers[1]), 0:1:length(mesh.centers[2]), 0:1:length(mesh.centers[3])) do vtk
                    vtk["Temperature_1_b"] = reshape(state[1:part], length(mesh.centers[1])+1, length(mesh.centers[2])+1, length(mesh.centers[3])+1)
                    vtk["Temperature_1_g"] = reshape(state[part + 1:2 * part], length(mesh.centers[1])+1, length(mesh.centers[2])+1, length(mesh.centers[3])+1)
                    vtk["Temperature_2_b"] = reshape(state[2 * part + 1:3 * part], length(mesh.centers[1])+1, length(mesh.centers[2])+1, length(mesh.centers[3])+1)
                    vtk["Temperature_2_g"] = reshape(state[3 * part + 1:end], length(mesh.centers[1])+1, length(mesh.centers[2])+1, length(mesh.centers[3])+1)
                    pvd[i] = vtk
                end
            end
            vtk_save(pvd)
            println("Fichier VTK unsteady diphasic écrit : $filename.pvd")
        else 
            error("La combinaison de TimeType, PhaseType et EquationType n'est pas supportée.")
        end
    else
        error("Invalid number of dimensions for mesh.centers.")
    end
end
