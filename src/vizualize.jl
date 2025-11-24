function plot_solution(solver, mesh::Mesh{1}, body::Function, capacity::Capacity; state_i=1)
    # Z_sdf = [body(x) for x in mesh.nodes[1]]

    # Déterminer le type de problème
    is_steady = solver.time_type == Steady # Problème stationnaire
    is_monophasic = solver.phase_type == Monophasic # Problème monophasique

    # Tracer selon la dimension et le type de problème
    if is_steady
        if is_monophasic # Monophasic
            uₒ = solver.x[1:length(solver.x) ÷ 2]
            uᵧ = solver.x[length(solver.x) ÷ 2 + 1:end]
            x = mesh.centers[1]

            # Désactiver les cellules désactivées
            cell_types = capacity.cell_types
            uₒ[cell_types .== 0] .= NaN
            #uᵧ[cell_types .== 0] .= NaN
            uᵧ[cell_types .== 1] .= NaN
            
            fig = Figure()
            ax = Axis(fig[1, 1], title="Monophasic Steady Solution", xlabel="x", ylabel="u")
            scatter!(ax, uₒ, color=:blue, label="Bulk")
            scatter!(ax, uᵧ, color=:green, label="Interface")
            axislegend(ax)
            display(fig)
        else # Diphasic
            u1ₒ = solver.x[1:length(solver.x) ÷ 4]
            u1ᵧ = solver.x[length(solver.x) ÷ 4 + 1:2*length(solver.x) ÷ 4]
            u2ₒ = solver.x[2*length(solver.x) ÷ 4 + 1:3*length(solver.x) ÷ 4]
            u2ᵧ = solver.x[3*length(solver.x) ÷ 4 + 1:end]

            # Désactiver les cellules désactivées
            cell_types = capacity.cell_types
            u1ₒ[cell_types .== 0] .= NaN
            u1ᵧ[cell_types .== 0] .= NaN
            u1ᵧ[cell_types .== 1] .= NaN
            u2ₒ[cell_types .== 1] .= NaN
            u2ᵧ[cell_types .== 1] .= NaN
            u2ᵧ[cell_types .== 0] .= NaN

            x = mesh.centers[1]
            fig = Figure()
            ax = Axis(fig[1, 1], title="Diphasic Steady Solutions", xlabel="x", ylabel="u")
            scatter!(ax, u1ₒ, color=:blue, label="Phase 1 - Bulk")
            scatter!(ax, u1ᵧ, color=:green, label="Phase 1 - Interface")
            scatter!(ax, u2ₒ, color=:red, label="Phase 2 - Bulk")
            scatter!(ax, u2ᵧ, color=:purple, label="Phase 2 - Interface")
            axislegend(ax, position=:rb)
            display(fig)
        end
    else
        # Tracé unsteady en 1D
        if is_monophasic # Monophasic
            states = solver.states
            fig = Figure()
            ax = Axis(fig[1, 1], title="Monophasic Unsteady Solutions", xlabel="x", ylabel="u")
            for state in states
                lines!(ax, state[1:length(state) ÷ 2], color=:blue, alpha=0.3, label="Bulk")
                lines!(ax, state[length(state) ÷ 2 + 1:end], color=:green, alpha=0.3, label="Interface")
            end
            #axislegend(ax)
            display(fig)
        else # Diphasic

            states1ₒ = solver.states[state_i][1:length(solver.states[state_i]) ÷ 4]  # Phase 1 - Bulk
            states1ᵧ = solver.states[state_i][length(solver.states[state_i]) ÷ 4 + 1:2*length(solver.states[state_i]) ÷ 4]  # Phase 1 - Interface
            states2ₒ = solver.states[state_i][2*length(solver.states[state_i]) ÷ 4 + 1:3*length(solver.states[state_i]) ÷ 4]  # Phase 2 - Bulk
            states2ᵧ = solver.states[state_i][3*length(solver.states[state_i]) ÷ 4 + 1:end]  # Phase 2 - Interface

            cell_types = capacity.cell_types
            states1ₒ[cell_types .== 0] .= NaN
            states1ᵧ[cell_types .== 0] .= NaN
            states1ᵧ[cell_types .== 1] .= NaN
            states2ₒ[cell_types .== 1] .= NaN
            states2ᵧ[cell_types .== 1] .= NaN
            states2ᵧ[cell_types .== 0] .= NaN

            fig = Figure(size=(800, 600))
            ax1 = Axis(fig[1, 1], title="Diphasic Unsteady - Phase 1", xlabel="x", ylabel="u1")
            ax2 = Axis(fig[2, 1], title="Diphasic Unsteady - Phase 2", xlabel="x", ylabel="u2")
            lines!(ax1, states1ₒ, color=:blue, label="Bulk")
            scatter!(ax1, states1ᵧ, color=:green, label="Interface")
            lines!(ax2, states2ₒ, color=:blue, label="Bulk")
            scatter!(ax2, states2ᵧ, color=:green, label="Interface")
            #axislegend(ax1)
            #axislegend(ax2)
            display(fig)
        end
    end
end

function plot_solution(solver, mesh::Mesh{2}, body::Function, capacity::Capacity; state_i=1)
    #Z_sdf = [body(x,y) for x in mesh.nodes[1], y in mesh.nodes[2]]
    is_steady = solver.time_type == Steady
    is_monophasic = solver.phase_type == Monophasic

    # Tracé en 2D
    if is_steady
        fig = Figure(size=(800, 600))
        if is_monophasic # Monophasic
            cell_types = capacity.cell_types
            uₒ = solver.x[1:length(solver.x) ÷ 2]
            uᵧ = solver.x[length(solver.x) ÷ 2 + 1:end]

            # Désactiver les cellules désactivées
            uₒ[cell_types .== 0] .= NaN
            uᵧ[cell_types .== 0] .= NaN
            uᵧ[cell_types .== 1] .= NaN

            reshaped_uₒ = reshape(uₒ, (length(mesh.centers[1])+1, length(mesh.centers[2])+1) )'
            reshaped_uᵧ = reshape(uᵧ, (length(mesh.centers[1])+1, length(mesh.centers[2])+1) )'

            ax1 = Axis(fig[1, 1], title="Monophasic Steady Solution - Bulk", xlabel="x", ylabel="y", aspect = DataAspect())
            ax2 = Axis(fig[1, 3], title="Monophasic Steady Solution - Interface", xlabel="x", ylabel="y", aspect = DataAspect())

            hm1 = heatmap!(ax1, mesh.centers[1], mesh.centers[2], reshaped_uₒ, colormap=:viridis)
            hm2 = heatmap!(ax2, mesh.centers[1], mesh.centers[2], reshaped_uᵧ, colormap=:viridis)

            #contour!(ax1, mesh.nodes[1], mesh.nodes[2], Z_sdf, levels=[0.0], color=:red, linewidth=2, label="SDF=0")
            #contour!(ax2, mesh.nodes[1], mesh.nodes[2], Z_sdf, levels=[0.0], color=:red, linewidth=2, label="SDF=0")

            Colorbar(fig[1, 2], hm1, label="Bulk Temperature")
            Colorbar(fig[1, 4], hm2, label="Interface Temperature")
        else # Diphasic
            cell_types = capacity.cell_types
            u1ₒ = solver.x[1:length(solver.x) ÷ 4]
            u1ᵧ = solver.x[length(solver.x) ÷ 4 + 1:2*length(solver.x) ÷ 4]
            u2ₒ = solver.x[2*length(solver.x) ÷ 4 + 1:3*length(solver.x) ÷ 4]
            u2ᵧ = solver.x[3*length(solver.x) ÷ 4+1:end]

            # Désactiver les cellules désactivées
            u1ₒ[cell_types .== 0] .= NaN
            u1ᵧ[cell_types .== 0] .= NaN
            u1ᵧ[cell_types .== 1] .= NaN
            u2ₒ[cell_types .== 1] .= NaN
            u2ᵧ[cell_types .== 1] .= NaN
            u2ᵧ[cell_types .== 0] .= NaN

            reshaped_u1ₒ = reshape(u1ₒ, (length(mesh.centers[1])+1, length(mesh.centers[2])+1) )'
            reshaped_u1ᵧ = reshape(u1ᵧ, (length(mesh.centers[1])+1, length(mesh.centers[2])+1) )'
            reshaped_u2ₒ = reshape(u2ₒ, (length(mesh.centers[1])+1, length(mesh.centers[2])+1) )'
            reshaped_u2ᵧ = reshape(u2ᵧ, (length(mesh.centers[1])+1, length(mesh.centers[2])+1) )'
            
            ax1 = Axis(fig[1, 1], title="Diphasic Steady - Phase 1 - Bulk", xlabel="x", ylabel="y", aspect = DataAspect())
            hm1 = heatmap!(ax1, mesh.centers[1], mesh.centers[2], reshaped_u1ₒ, colormap=:viridis)
            cb1 = Colorbar(fig[1, 2], hm1, label="Phase 1 Bulk Temperature")
            #contour!(ax1, mesh.nodes[1], mesh.nodes[2], Z_sdf, levels=[0.0], color=:red, linewidth=2, label="SDF=0")

            ax2 = Axis(fig[1, 3], title="Diphasic Steady - Phase 1 - Interface", xlabel="x", ylabel="y", aspect = DataAspect())
            hm2 = heatmap!(ax2, mesh.centers[1], mesh.centers[2], reshaped_u1ᵧ, colormap=:viridis)
            cb2 = Colorbar(fig[1, 4], hm2, label="Phase 1 Interface Temperature")
            #contour!(ax2, mesh.nodes[1], mesh.nodes[2], Z_sdf, levels=[0.0], color=:red, linewidth=2, label="SDF=0")

            ax3 = Axis(fig[2, 1], title="Diphasic Steady - Phase 2 - Bulk", xlabel="x", ylabel="y", aspect = DataAspect())
            hm3 = heatmap!(ax3, mesh.centers[1], mesh.centers[2], reshaped_u2ₒ, colormap=:viridis)
            cb3 = Colorbar(fig[2, 2], hm3, label="Phase 2 Bulk Temperature")
            #contour!(ax3, mesh.nodes[1], mesh.nodes[2], Z_sdf, levels=[0.0], color=:red, linewidth=2, label="SDF=0")

            ax4 = Axis(fig[2, 3], title="Diphasic Steady - Phase 2 - Interface", xlabel="x", ylabel="y", aspect = DataAspect())
            hm4 = heatmap!(ax4, mesh.centers[1], mesh.centers[2], reshaped_u2ᵧ, colormap=:viridis)
            cb4 = Colorbar(fig[2, 4], hm4, label="Phase 2 Interface Temperature")
            #contour!(ax4, mesh.nodes[1], mesh.nodes[2], Z_sdf, levels=[0.0], color=:red, linewidth=2, label="SDF=0")

        end
        display(fig)
    else
        # Tracé unsteady en 2D
        if is_monophasic
            cell_types = capacity.cell_types
            uₒ = solver.states[state_i][1:length(solver.states[1]) ÷ 2]
            uᵧ = solver.states[state_i][length(solver.states[1]) ÷ 2 + 1:end]

            # Désactiver les cellules désactivées
            uₒ[cell_types .== 0] .= NaN
            uᵧ[cell_types .== 0] .= NaN
            uᵧ[cell_types .== 1] .= NaN

            reshaped_uₒ = reshape(uₒ, (length(mesh.centers[1])+1, length(mesh.centers[2])+1) )'
            reshaped_uᵧ = reshape(uᵧ, (length(mesh.centers[1])+1, length(mesh.centers[2])+1) )'

            fig = Figure(size=(800, 600))

            ax1 = Axis(fig[1, 1], title="Monophasic Unsteady Diffusion - Bulk", xlabel="x", ylabel="y", aspect = DataAspect())
            hm1 = heatmap!(ax1, mesh.centers[1], mesh.centers[2], reshaped_uₒ, colormap=:viridis)
            #contour!(ax1, mesh.nodes[1], mesh.nodes[2], Z_sdf, levels=[0.0], color=:red, linewidth=2, label="SDF=0")
            Colorbar(fig[1, 2], hm1, label="Bulk Temperature")

            ax2 = Axis(fig[1, 3], title="Monophasic Unsteady Diffusion - Interface", xlabel="x", ylabel="y", aspect = DataAspect())
            hm2 = heatmap!(ax2, mesh.centers[1], mesh.centers[2], reshaped_uᵧ, colormap=:viridis)
            #contour!(ax2, mesh.nodes[1], mesh.nodes[2], Z_sdf, levels=[0.0], color=:red, linewidth=2, label="SDF=0")
            Colorbar(fig[1, 4], hm2, label="Interface Temperature")

            display(fig)  
        else
            # Plot Last State
            cell_types = capacity.cell_types
            u1ₒ = solver.states[state_i][1:length(solver.states[1]) ÷ 4]
            u1ᵧ = solver.states[state_i][length(solver.states[1]) ÷ 4 + 1:2*length(solver.states[1]) ÷ 4]
            u2ₒ = solver.states[state_i][2*length(solver.states[1]) ÷ 4 + 1:3*length(solver.states[1]) ÷ 4]
            u2ᵧ = solver.states[state_i][3*length(solver.states[1]) ÷ 4 + 1:end]

            # Désactiver les cellules désactivées
            u1ₒ[cell_types .== 0] .= NaN
            u1ᵧ[cell_types .== 0] .= NaN
            u1ᵧ[cell_types .== 1] .= NaN
            u2ₒ[cell_types .== 1] .= NaN
            u2ᵧ[cell_types .== 1] .= NaN
            u2ᵧ[cell_types .== 0] .= NaN

            reshaped_u1ₒ = reshape(u1ₒ, (length(mesh.centers[1])+1, length(mesh.centers[2])+1) )'
            reshaped_u1ᵧ = reshape(u1ᵧ, (length(mesh.centers[1])+1, length(mesh.centers[2])+1) )'
            reshaped_u2ₒ = reshape(u2ₒ, (length(mesh.centers[1])+1, length(mesh.centers[2])+1) )'
            reshaped_u2ᵧ = reshape(u2ᵧ, (length(mesh.centers[1])+1, length(mesh.centers[2])+1) )'

            fig = Figure(size=(800, 600))

            x, y = mesh.centers[1], mesh.centers[2]

            ax1 = Axis(fig[1, 1], title="Diphasic Unsteady - Phase 1 - Bulk", xlabel="x", ylabel="y", aspect = DataAspect())
            hm1 = heatmap!(ax1, x, y, reshaped_u1ₒ, colormap=:viridis)
            Colorbar(fig[1, 2], hm1, label="Phase 1 Bulk Temperature")

            ax2 = Axis(fig[1, 3], title="Diphasic Unsteady - Phase 1 - Interface", xlabel="x", ylabel="y", aspect = DataAspect())
            hm2 = heatmap!(ax2, x, y, reshaped_u1ᵧ, colormap=:viridis)
            Colorbar(fig[1, 4], hm2, label="Phase 1 Interface Temperature")

            ax3 = Axis(fig[2, 1], title="Diphasic Unsteady - Phase 2 - Bulk", xlabel="x", ylabel="y", aspect = DataAspect())
            hm3 = heatmap!(ax3, x, y, reshaped_u2ₒ, colormap=:viridis)
            Colorbar(fig[2, 2], hm3, label="Phase 2 Bulk Temperature")

            ax4 = Axis(fig[2, 3], title="Diphasic Unsteady - Phase 2 - Interface", xlabel="x", ylabel="y", aspect = DataAspect())
            hm4 = heatmap!(ax4, x, y, reshaped_u2ᵧ, colormap=:viridis)
            Colorbar(fig[2, 4], hm4, label="Phase 2 Interface Temperature")

            display(fig)
        end
    end
end


function plot_solution(solver, mesh::Mesh{3}, body::Function, capacity::Capacity; state_i=1)
    #Z_sdf = [body.sdf(xi, yi, zi) for zi in mesh.nodes[3], yi in mesh.nodes[2], xi in mesh.nodes[1]]

    is_steady = solver.time_type == Steady
    is_monophasic = solver.phase_type == Monophasic

    # Tracé en 3D
    if is_steady
        if is_monophasic
            fig = Figure()
            ax = LScene(fig[1, 1], show_axis=false)

            nx, ny, nz = length(mesh.centers[1]), length(mesh.centers[2]), length(mesh.centers[3])
            x = mesh.centers[1]
            y = mesh.centers[2]
            z = mesh.centers[3]

            sgrid = SliderGrid(
                fig[2, 1],
                (label = "yz plane - x axis", range = 1:length(x)),
                (label = "xz plane - y axis", range = 1:length(y)),
                (label = "xy plane - z axis", range = 1:length(z)),
            )

            lo = sgrid.layout
            nc = ncols(lo)

            vol = solver.x[1:length(solver.x) ÷ 2]
            vol = reshape(vol, nx+1, ny+1, nz+1)
            plt = volumeslices!(ax, x, y, z, vol)
            Colorbar(fig[1, 2], plt)

            # connect sliders to `volumeslices` update methods
            sl_yz, sl_xz, sl_xy = sgrid.sliders

            on(sl_yz.value) do v; plt[:update_yz][](v) end
            on(sl_xz.value) do v; plt[:update_xz][](v) end
            on(sl_xy.value) do v; plt[:update_xy][](v) end

            set_close_to!(sl_yz, .5length(x))
            set_close_to!(sl_xz, .5length(y))
            set_close_to!(sl_xy, .5length(z))

            # add toggles to show/hide heatmaps
            hmaps = [plt[Symbol(:heatmap_, s)][] for s ∈ (:yz, :xz, :xy)]
            toggles = [Toggle(lo[i, nc + 1], active = true) for i ∈ 1:length(hmaps)]

            map(zip(hmaps, toggles)) do (h, t)
                connect!(h.visible, t.active)
            end

            # cam3d!(ax.scene, projectiontype=Makie.Orthographic)

            display(fig)
        else # Diphasic
            fig = Figure()
            ax1 = LScene(fig[1, 1], show_axis=false)
            ax2 = LScene(fig[1, 3], show_axis=false)

            nx, ny, nz = length(mesh.centers[1]), length(mesh.centers[2]), length(mesh.centers[3])
            x = LinRange(mesh.x0[1], mesh.x0[1]+mesh.h[1][1]*nx, nx+1)
            y = LinRange(mesh.x0[2], mesh.x0[2]+mesh.h[2][1]*ny, ny+1)
            z = LinRange(mesh.x0[3], mesh.x0[3]+mesh.h[3][1]*nz, nz+1)

            sgrid = SliderGrid(
                fig[2, 1],
                (label = "yz plane - x axis", range = 1:length(x)),
                (label = "xz plane - y axis", range = 1:length(y)),
                (label = "xy plane - z axis", range = 1:length(z)),
            )

            lo = sgrid.layout
            nc = ncols(lo)

            vol1 = solver.x[1:length(solver.x) ÷ 4]
            vol1 = reshape(vol1, nx+1, ny+1, nz+1)
            plt1 = volumeslices!(ax1, x, y, z, vol1)
            Colorbar(fig[1, 2], plt1)

            vol2 = solver.x[2*length(solver.x) ÷ 4 + 1:3*length(solver.x) ÷ 4]
            vol2 = reshape(vol2, nx+1, ny+1, nz+1)
            plt2 = volumeslices!(ax2, x, y, z, vol2)
            Colorbar(fig[1, 4], plt2)

            # connect sliders to `volumeslices` update methods
            sl_yz, sl_xz, sl_xy = sgrid.sliders

            on(sl_yz.value) do v
                plt1[:update_yz][](v) 
                plt2[:update_yz][](v)
            end
            on(sl_xz.value) do v
                plt1[:update_xz][](v)
                plt2[:update_xz][](v)
            end
            on(sl_xy.value) do v
                plt1[:update_xy][](v)
                plt2[:update_xy][](v)
            end

            set_close_to!(sl_yz, .5length(x))
            set_close_to!(sl_xz, .5length(y))
            set_close_to!(sl_xy, .5length(z))

            # add toggles to show/hide heatmaps
            hmaps = [plt1[Symbol(:heatmap_, s)][] for s ∈ (:yz, :xz, :xy)]
            toggles = [Toggle(lo[i, nc + 1], active = true) for i ∈ 1:length(hmaps)]

            map(zip(hmaps, toggles)) do (h, t)
                connect!(h.visible, t.active)
            end

            # cam3d!(ax1.scene, projectiontype=Makie.Orthographic)
            # cam3d!(ax2.scene, projectiontype=Makie.Orthographic)
            display(fig)
        end
    else # Unsteady
        if is_monophasic
            fig = Figure()
            ax = LScene(fig[1, 1], show_axis=false)

            nx, ny, nz = length(mesh.centers[1]), length(mesh.centers[2]), length(mesh.centers[3])
            x = LinRange(mesh.x0[1], mesh.x0[1]+mesh.h[1][1]*nx, nx+1)
            y = LinRange(mesh.x0[2], mesh.x0[2]+mesh.h[2][1]*ny, ny+1)
            z = LinRange(mesh.x0[3], mesh.x0[3]+mesh.h[3][1]*nz, nz+1)

            sgrid = SliderGrid(
                fig[2, 1],
                (label = "yz plane - x axis", range = 1:length(x)),
                (label = "xz plane - y axis", range = 1:length(y)),
                (label = "xy plane - z axis", range = 1:length(z)),
            )

            lo = sgrid.layout
            nc = ncols(lo)

            vol = solver.states[state_i][1:length(solver.states[state_i]) ÷ 2]
            vol = reshape(vol, nx+1, ny+1, nz+1)
            plt = volumeslices!(ax, x, y, z, vol)
            Colorbar(fig[1, 2], plt)

            # connect sliders to `volumeslices` update methods
            sl_yz, sl_xz, sl_xy = sgrid.sliders

            on(sl_yz.value) do v; plt[:update_yz][](v) end
            on(sl_xz.value) do v; plt[:update_xz][](v) end
            on(sl_xy.value) do v; plt[:update_xy][](v) end

            set_close_to!(sl_yz, .5length(x))
            set_close_to!(sl_xz, .5length(y))
            set_close_to!(sl_xy, .5length(z))

            # add toggles to show/hide heatmaps
            hmaps = [plt[Symbol(:heatmap_, s)][] for s ∈ (:yz, :xz, :xy)]
            toggles = [Toggle(lo[i, nc + 1], active = true) for i ∈ 1:length(hmaps)]

            map(zip(hmaps, toggles)) do (h, t)
                connect!(h.visible, t.active)
            end

            # cam3d!(ax.scene, projectiontype=Makie.Orthographic)  

            display(fig)

        else # Diphasic
            fig = Figure()
            ax1 = LScene(fig[1, 1], show_axis=false)
            ax2 = LScene(fig[1, 3], show_axis=false)

            nx, ny, nz = length(mesh.centers[1]), length(mesh.centers[2]), length(mesh.centers[3])
            x = LinRange(mesh.x0[1], mesh.x0[1]+mesh.h[1][1]*nx, nx+1)
            y = LinRange(mesh.x0[2], mesh.x0[2]+mesh.h[2][1]*ny, ny+1)
            z = LinRange(mesh.x0[3], mesh.x0[3]+mesh.h[3][1]*nz, nz+1)

            sgrid = SliderGrid(
                fig[2, 1],
                (label = "yz plane - x axis", range = 1:length(x)),
                (label = "xz plane - y axis", range = 1:length(y)),
                (label = "xy plane - z axis", range = 1:length(z)),
            )

            lo = sgrid.layout
            nc = ncols(lo)

            vol1 = solver.states[state_i][1:length(solver.states[state_i]) ÷ 4]
            vol1 = reshape(vol1, nx+1, ny+1, nz+1)
            plt1 = volumeslices!(ax1, x, y, z, vol1)
            Colorbar(fig[1, 2], plt1)

            vol2 = solver.states[state_i][2*length(solver.states[state_i]) ÷ 4 + 1:3*length(solver.states[state_i]) ÷ 4]
            vol2 = reshape(vol2, nx+1, ny+1, nz+1)
            plt2 = volumeslices!(ax2, x, y, z, vol2)
            Colorbar(fig[1, 4], plt2)

            # connect sliders to `volumeslices` update methods
            sl_yz, sl_xz, sl_xy = sgrid.sliders

            on(sl_yz.value) do v
                plt1[:update_yz][](v) 
                plt2[:update_yz][](v)
            end
            on(sl_xz.value) do v
                plt1[:update_xz][](v)
                plt2[:update_xz][](v)
            end
            on(sl_xy.value) do v
                plt1[:update_xy][](v)
                plt2[:update_xy][](v)
            end

            set_close_to!(sl_yz, .5length(x))
            set_close_to!(sl_xz, .5length(y))
            set_close_to!(sl_xy, .5length(z))

            # add toggles to show/hide heatmaps
            hmaps = [plt1[Symbol(:heatmap_, s)][] for s ∈ (:yz, :xz, :xy)]
            toggles = [Toggle(lo[i, nc + 1], active = true) for i ∈ 1:length(hmaps)]

            map(zip(hmaps, toggles)) do (h, t)
                connect!(h.visible, t.active)
            end

            # cam3d!(ax1.scene, projectiontype=Makie.Orthographic)
            # cam3d!(ax2.scene, projectiontype=Makie.Orthographic)
            display(fig)
        end
    end
end

"""
    animate_solution(solver, mesh, body)

Routine to animate the solution once the solver has been run.

# Arguments
- `solver::Solver`: The solver of the problem.
- `mesh::Mesh`: The mesh of the problem.
- `body::Function`: The body function of the problem.
"""
function animate_solution(solver, mesh::Mesh{1}, body::Function)
    # Déterminer le type de problème
    is_monophasic = solver.phase_type == Monophasic # Problème monophasique

    # Enregistrer l'animation selon le type de problème
    if is_monophasic
        # Récupérer les états
        states = solver.states

        # Créer une figure
        fig = Figure()

        # Créer un axe pour la figure
        ax = Axis(fig[1, 1], title="Monophasic Unsteady Diffusion", xlabel="x", ylabel="u")
    
        x = mesh.nodes[1]
        ylims!(ax, (minimum([minimum(state[1:length(state) ÷ 2]) for state in states]), maximum([maximum(state[1:length(state) ÷ 2]) for state in states])))
        xlims!(ax, (minimum(x), maximum(x)))
        function update_ln(frame)
            # Récupérer l'état
            state = states[frame]

            # Tracer l'état
            lines!(ax, x, state[1:length(state) ÷ 2], color=:blue, alpha=0.3, label="Bulk")
            lines!(ax, x, state[length(state) ÷ 2 + 1:end], color=:green, alpha=0.3, label="Interface")
        end

        # Enregistrer l'animation
        record(fig, "heat_MonoUnsteady.mp4", 1:length(states); framerate=10) do frame
            update_ln(frame)
        end

        # Afficher la figure
        display(fig)
    else
        # Récupérer les états
        states = solver.states

        # Créer une figure
        fig = Figure()

        # Créer un axe pour la figure
        ax1 = Axis(fig[1, 1], title="Diphasic Unsteady - Phase 1", xlabel="x", ylabel="u1")
        ax2 = Axis(fig[2, 1], title="Diphasic Unsteady - Phase 2", xlabel="x", ylabel="u2")

        x = mesh.nodes[1]
        ylims!(ax1, (0.0, 1.0))
        ylims!(ax2, (0.0, 1.0))
        xlims!(ax1, (minimum(x), maximum(x)))
        xlims!(ax2, (minimum(x), maximum(x)))
        # Créer une fonction pour mettre à jour la figure
        function update_plot(frame)
            # Récupérer l'état
            state = states[frame]

            # Tracer l'état
            lines!(ax1, x, state[1:length(state) ÷ 4], color=:blue, alpha=0.3, label="Bulk")
            lines!(ax1, x, state[length(state) ÷ 4 + 1:2*length(state) ÷ 4], color=:green, alpha=0.3, label="Interface")
            lines!(ax2, x, state[2*length(state) ÷ 4 + 1:3*length(state) ÷ 4], color=:blue, alpha=0.3, label="Bulk")
            lines!(ax2, x, state[3*length(state) ÷ 4 + 1:end], color=:green, alpha=0.3, label="Interface")

        end

        # Enregistrer l'animation
        record(fig, "heat_DiphUnsteady.mp4", 1:length(states); framerate=10) do frame
            update_plot(frame)
        end

        # Afficher la figure
        display(fig)

    end
end


function animate_solution(solver, mesh::Mesh{2}, body::Function)
    # Déterminer le type de problème
    is_monophasic = solver.phase_type == Monophasic # Problème monophasique

    # Enregistrer l'animation selon le type de problème
    if is_monophasic
        # Récupérer les états
        states = solver.states

        # Créer une figure
        fig = Figure()

        # Créer un axe pour la figure
        ax = Axis(fig[1, 1], title="Monophasic Unsteady", xlabel="x", ylabel="y", aspect=DataAspect())
        
        x,y = mesh.nodes[1], mesh.nodes[2]
        min_val = minimum([minimum(reshape(state[1:length(state) ÷ 2], (length(mesh.centers[1])+1, length(mesh.centers[2])+1))') for state in solver.states])
        max_val = maximum([maximum(reshape(state[1:length(state) ÷ 2], (length(mesh.centers[1])+1, length(mesh.centers[2])+1))') for state in solver.states])

        hm = heatmap!(ax, x, y, reshape(states[1][1:length(states[1]) ÷ 2], (length(mesh.centers[1])+1, length(mesh.centers[2])+1))', colormap=:viridis, colorrange=(min_val, max_val))
        Colorbar(fig[1, 2], hm, label="Temperature")

        # In your update function for the heatmap, ensure that you convert the transposed array 
        # (an Adjoint) into a standard Matrix by using collect().
        update_hm(frame) = collect(
            reshape(
                solver.states[frame][1:length(solver.states[frame]) ÷ 2],
                (length(mesh.centers[1])+1, length(mesh.centers[2])+1)
            )'
        )
        
        # Then in your record block, update the heatmap as follows:
        record(fig, "heat_MonoUnsteady.mp4", 1:length(solver.states); framerate=10) do frame
            #hm[1] = update_hm(frame)
            heatmap!(ax, x, y, update_hm(frame), colormap=:viridis, colorrange=(min_val, max_val))
        end


        display(fig)
    else
        # Récupérer les états
        states = solver.states

        # Créer une figure
        fig = Figure(size=(800, 400))

        # Créer un axe pour la figure
        ax1 = Axis3(fig[1, 1], title="Diphasic Unsteady - Phase 1 - Bulk", xlabel="x", ylabel="y", zlabel="Temperature")
        s1 = surface!(ax1, reshape(states[1][1:length(states[1]) ÷ 4], (length(mesh.centers[1])+1, length(mesh.centers[2])+1))', colormap=:viridis)

        ax2 = Axis3(fig[1, 2], title="Diphasic Unsteady - Phase 1 - Interface", xlabel="x", ylabel="y", zlabel="Temperature")
        s2 = surface!(ax2, reshape(states[1][length(states[1]) ÷ 4 + 1:2*length(states[1]) ÷ 4], (length(mesh.centers[1])+1, length(mesh.centers[2])+1))', colormap=:viridis)

        ax3 = Axis3(fig[2, 1], title="Diphasic Unsteady - Phase 2 - Bulk", xlabel="x", ylabel="y", zlabel="Temperature")
        s3 = surface!(ax3, reshape(states[1][2*length(states[1]) ÷ 4 + 1:3*length(states[1]) ÷ 4], (length(mesh.centers[1])+1, length(mesh.centers[2])+1))', colormap=:viridis)

        ax4 = Axis3(fig[2, 2], title="Diphasic Unsteady - Phase 2 - Interface", xlabel="x", ylabel="y", zlabel="Temperature")
        s4 = surface!(ax4, reshape(states[1][3*length(states[1]) ÷ 4 + 1:end], (length(mesh.centers[1])+1, length(mesh.centers[2])+1))', colormap=:viridis)

        zlims!(ax1, 0, 1)
        zlims!(ax2, 0, 1)
        zlims!(ax3, 0, 1)
        zlims!(ax4, 0, 1)

        function update_surfaces!(frame)
            s1[:z] = reshape(solver.states[frame][1:length(solver.states[frame]) ÷ 4], (length(mesh.centers[1])+1, length(mesh.centers[2])+1))'
            s2[:z] = reshape(solver.states[frame][length(solver.states[frame]) ÷ 4 + 1:2*length(solver.states[frame]) ÷ 4], (length(mesh.centers[1])+1, length(mesh.centers[2])+1))'
            s3[:z] = reshape(solver.states[frame][2*length(solver.states[frame]) ÷ 4 + 1:3*length(solver.states[frame]) ÷ 4], (length(mesh.centers[1])+1, length(mesh.centers[2])+1))'
            s4[:z] = reshape(solver.states[frame][3*length(solver.states[frame]) ÷ 4 + 1:end], (length(mesh.centers[1])+1, length(mesh.centers[2])+1))'
            println("Frame $frame")
        end

        record(fig, "heat_DiphUnsteady.mp4", 1:length(solver.states); framerate=10) do frame
            update_surfaces!(frame)
        end

        display(fig)
    end
end

"""
    plot_isotherms(solver, mesh, nx, ny; 
                   time_steps=nothing, 
                   levels=[0.01, 0.05, 0.1, 0.2, 0.5, 0.8], 
                   layout=(2,2), 
                   colormap=:viridis,
                   show_interface=true,
                   interpolate=false)

Trace les isothermes (lignes d'égale température) à différents pas de temps.

Paramètres:
- `solver` : Solveur contenant les états à différents pas de temps
- `mesh` : Maillage utilisé
- `nx`, `ny` : Dimensions du maillage 
- `time_steps` : Indices des pas de temps à visualiser (par défaut : 4 répartis uniformément)
- `levels` : Valeurs des isothermes à tracer
- `layout` : Disposition des sous-graphiques (lignes, colonnes)
- `colormap` : Palette de couleurs pour les isothermes
- `show_interface` : Si vrai, affiche la position de l'interface
- `interpolate` : Si vrai, interpole les données sur une grille plus fine

Retourne:
- Une figure CairoMakie
"""
function plot_isotherms(solver, mesh, nx, ny, reconstruct; 
                        time_steps=nothing, 
                        levels=[0.01, 0.05, 0.1, 0.2, 0.5, 0.8], 
                        layout=(2,2), 
                        colormap=:viridis,
                        show_interface=true,
                        interpolate=false)
    
    # Déterminer les pas de temps à afficher
    num_states = length(solver.states)
    if time_steps === nothing
        # Sélectionner automatiquement des pas de temps bien répartis
        steps_to_show = min(prod(layout), num_states)
        time_steps = round.(Int, range(1, num_states, length=steps_to_show))
    end
    
    # S'assurer que time_steps ne dépasse pas le nombre d'états disponibles
    time_steps = filter(t -> 1 <= t <= num_states, time_steps)
    
    # Créer la figure
    fig = Figure(resolution=(800*layout[2], 700*layout[1]))
    
    # Coordonnées spatiales pour le traçage
    x_coords = mesh.nodes[1][1:end-1]
    y_coords = mesh.nodes[2][1:end-1]
    
    # Pour chaque pas de temps sélectionné
    for (i, step) in enumerate(time_steps)
        # Calculer la position dans la grille de sous-graphiques
        row = (i - 1) ÷ layout[2] + 1
        col = (i - 1) % layout[2] + 1
        
        # Créer l'axe pour ce pas de temps
        ax = Axis(fig[row, col], 
                 xlabel="x", 
                 ylabel="y", 
                 title="Isothermes - Pas de temps $step",
                 aspect=DataAspect())
        
        # Extraire le champ de température pour ce pas de temps
        T_vec = solver.states[step][1:((nx+1)*(ny+1))]
        T_field = reshape(T_vec, (nx+1, ny+1))
        T_field = T_field'  # Transposer pour l'orientation correcte
        T_field = Float64.(T_field)
        
        # Ajuster la taille pour correspondre au maillage
        T_field = T_field[1:end-1, 1:end-1]
        
        # Interpoler sur une grille plus fine si demandé
        if interpolate
            # Créer une grille plus fine
            x_fine = range(minimum(x_coords), maximum(x_coords), length=3*length(x_coords))
            y_fine = range(minimum(y_coords), maximum(y_coords), length=3*length(y_coords))
            
            # Interpoler les données
            itp = linear_interpolation((x_coords, y_coords), T_field)
            T_fine = [itp(x,y) for x in x_fine, y in y_fine]
            
            # Tracer les contours avec les données interpolées
            contourf!(ax, x_fine, y_fine, T_fine, levels=levels, colormap=colormap)
            contour!(ax, x_fine, y_fine, T_fine, levels=levels, color=:black, linewidth=1.5)
        else
            # Tracer les contours directement
            contourf!(ax, x_coords, y_coords, T_field, levels=levels, colormap=colormap)
            contour!(ax, x_coords, y_coords, T_field, levels=levels, color=:black, linewidth=1.5)
        end
        
        # Ajouter l'interface si demandé
        if show_interface && step <= length(solver.states) && step <= length(reconstruct)
            # Obtenir la fonction d'interface pour ce pas de temps
            s_func = reconstruct[step]
            
            # Tracer l'interface
            y_vals = range(minimum(mesh.nodes[2]), maximum(mesh.nodes[2]), length=100)
            x_vals = s_func.(y_vals)
            
            # Ajouter la ligne d'interface
            lines!(ax, y_vals, x_vals, color=:red, linewidth=3, label="Interface")
        end
        
        # Limites des axes
        ax.limits = (minimum(x_coords), maximum(x_coords), minimum(y_coords), maximum(y_coords))
    end
    
    # Ajouter une barre de couleur commune
    Colorbar(fig[:, layout[2]+1], colormap=colormap, limits=(minimum(levels), maximum(levels)), 
             label="Température")
    
    # Légende commune si l'interface est affichée
    if show_interface
        Legend(fig[layout[1]+1, :], [LineElement(color=:red, linewidth=3)], 
               ["Interface"], orientation=:horizontal)
    end
    
    
    return fig
end