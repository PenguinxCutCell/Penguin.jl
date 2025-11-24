function lin_interpol(x_mesh, H_values; extrapolate=true)
    nx = length(H_values)
    N = 2*nx
    
    # Points aux interfaces et aux centres
    x = x_mesh
    dx = x[2] - x[1]
    
    # Valeurs aux interfaces (à déterminer)
    M = zeros(N, N)
    b = zeros(N)

    function idx(i, locale)
        return 2*(i-1) + (locale+1)
    end

    for i in 1:nx
        ip1 = (i == nx) ? 1 : i + 1

        # 1) Volume eq : H_i * dx = ∫(ai + bi*x) dx from 0 to dx
        #    This expands to: H_i * dx = ai*dx + bi*dx^2/2
        #    Dividing by dx: H_i = ai + bi*dx/2
        rowV = 2*(i-1) + 1
        M[rowV, idx(i,0)] = 1.0        # a_i
        M[rowV, idx(i,1)] = 0.5*dx     # b_i * dx/2
        b[rowV] = H_values[i]

        # 2) Value continuity eq : a_i + b_i*dx = a_{i+1}
        rowC = 2*(i-1) + 2
        M[rowC, idx(i,0)] = 1.0      # a_i
        M[rowC, idx(i,1)] = dx       # b_i * dx
        M[rowC, idx(ip1,0)] = -1.0   # -a_{i+1}
        b[rowC] = 0.0
    end

    # We have N-1 equations so far, we need one more for periodicity
    # Replace one of the continuity equations with periodicity: b_1 = b_{nx}, a_1 = a_{nx}
    M[N-1, idx(1,0)] = 1.0
    M[N-1, idx(nx,0)] = -1.0
    b[N-1] = 0.0

    M[N, idx(1,1)] = 1.0
    M[N, idx(nx,1)] = -1.0
    b[N] = 0.0

    s = M \ b

    a, b = s[1:2:end], s[2:2:end]

    # Fonction d'interpolation avec extrapolation
    function h_tilde(x_val)
        # Gestion de l'extrapolation
        if extrapolate
            # Si x_val est en dehors du domaine, extrapolation linéaire
            if x_val < x[1]
                # Extrapolation à gauche en utilisant la pente du premier segment
                return a[1] + b[1] * (x_val - x[1])
            elseif x_val > x[end]
                # Extrapolation à droite en utilisant la pente du dernier segment
                return a[nx] + b[nx] * (x_val - x[nx])
            end
        end
        
        # Trouver dans quelle cellule se trouve x_val (interpolation normale)
        i = 1
        while i < nx && !(x[i] <= x_val && x_val <= x[i+1])
            i += 1
        end
        
        # Si x_val est en dehors du domaine et que l'extrapolation est désactivée
        if (i >= nx && x_val > x[end]) || (i < 1 && x_val < x[1])
            if !extrapolate
                return 0.0  # En dehors du domaine sans extrapolation
            end
        end
        
        # Interpolation linéaire entre les interfaces
        xi = x_val - x[i]
        return a[i] + b[i] * xi
    end

    return h_tilde
end

# Quadratic interpolation function
function quad_interpol(x_mesh, H; extrapolate=true)
    nx = length(H)
    N = 3*nx
    
    # Points aux interfaces et aux centres
    x = x_mesh
    dx = x[2] - x[1]
    
    # Valeurs aux interfaces (à déterminer)
    # We need 3*nx equations for 3*nx unknowns
    # We have 3 equations per cell (nx cells), but need to replace two equations for BCs
    A = zeros(N, N)
    rhs = zeros(N)

    function idx(i, locale)
        return 3*(i-1) + (locale+1)
    end

    for i in 1:nx
        ip1 = (i == nx) ? 1 : i + 1

        # 1) Volume eq : H_i = a_i + 0.5 Δx b_i + 1/3 Δx^2 c_i
        rowV = 3*(i-1) + 1
        A[rowV, idx(i,0)] = 1.0        # a_i
        A[rowV, idx(i,1)] = 0.5*dx     # b_i
        A[rowV, idx(i,2)] = 1/3*dx^2   # c_i
        rhs[rowV] = H[i]

        # 2) Value continuity eq : a_i + b_i Δx + c_i Δx^2 = a_{i+1}
        rowC1 = 3*(i-1) + 2
        A[rowC1, idx(i,0)] = 1.0      # a_i
        A[rowC1, idx(i,1)] = dx       # b_i
        A[rowC1, idx(i,2)] = dx^2     # c_i
        A[rowC1, idx(ip1,0)] = -1.0   # -a_{i+1}
        rhs[rowC1] = 0.0

        # 3) Derivative continuity eq : b_i + 2 c_i Δx = b_{i+1}
        rowC2 = 3*(i-1) + 3
        A[rowC2, idx(i,1)] = 1.0      # b_i
        A[rowC2, idx(i,2)] = 2.0*dx   # c_i
        A[rowC2, idx(ip1,1)] = -1.0   # -b_{i+1}
        rhs[rowC2] = 0.0
    end
    
    # 4) BC : a_1 = a_{n_x}, b_1 = b_{n_x}, c_1 = c_{n_x} (periodicity)
    # Replace the last two equations for boundary conditions
    A[N-1, idx(1,0)] = 1.0
    A[N-1, idx(nx,0)] = -1.0
    rhs[N-1] = 0.0

    A[N, idx(1,1)] = 1.0
    A[N, idx(nx,1)] = -1.0
    rhs[N] = 0.0

    s = A \ rhs

    a, b, c = s[1:3:end], s[2:3:end], s[3:3:end]

    # Fonction d'interpolation avec extrapolation
    function h_tilde(x_val)
        # Gestion de l'extrapolation
        if extrapolate
            # Si x_val est en dehors du domaine
            if x_val < x[1]
                # Extrapolation quadratique à gauche
                # Utiliser le polynôme de la première cellule
                xi = x_val - x[1]
                return a[1] + b[1]*xi + c[1]*xi^2
            elseif x_val > x[end]
                # Extrapolation quadratique à droite
                # Utiliser le polynôme de la dernière cellule
                xi = x_val - x[nx]
                return a[nx] + b[nx]*xi + c[nx]*xi^2
            end
        end
        
        # Trouver dans quelle cellule se trouve x_val (interpolation normale)
        i = 1
        while i < nx && !(x[i] <= x_val && x_val < x[i+1])
            i += 1
        end
        
        # Gestion des cas limites
        if i > nx || i < 1
            if !extrapolate
                return 0.0  # En dehors du domaine sans extrapolation
            else
                # Ce cas ne devrait pas se produire normalement car traité plus haut
                # Mais gardons-le pour la robustesse
                if i > nx
                    i = nx
                    xi = x_val - x[i]
                else
                    i = 1
                    xi = x_val - x[i]
                end
                return a[i] + b[i]*xi + c[i]*xi^2
            end
        end
        
        # Cas spécial pour le dernier point qui peut être exactement égal à x[end]
        if i == nx && x_val == x[end]
            xi = x_val - x[nx]
            return a[nx] + b[nx]*xi + c[nx]*xi^2
        end
        
        # Interpolation quadratique dans la cellule
        xi = x_val - x[i]
        return a[i] + b[i]*xi + c[i]*xi^2
    end
    
    return h_tilde
end

# Cubic interpolation function
function cubic_interpol(x_mesh, H; extrapolate=true)
    nx = length(H)
    N = 4*nx
    
    # Points aux interfaces
    x = x_mesh
    dx = x[2] - x[1]

    # Valeurs aux interfaces (à déterminer)
    # We need 4*nx equations for 4*nx unknowns
    # We have 4 equations per cell (nx cells), but need to replace three equations for BCs
    A = zeros(N, N)
    rhs = zeros(N)

    function idx(i, locale)
        return 4*(i-1) + (locale+1)
    end

    for i in 1:nx
        ip1 = (i == nx) ? 1 : i + 1

        # 1) Volume eq : H_i = a_i + 0.5 Δx b_i + 1/3 Δx^2 c_i + 0.25 Δx^3 d_i
        rowV = 4*(i-1) + 1
        A[rowV, idx(i,0)] = 1.0        # a_i
        A[rowV, idx(i,1)] = 0.5*dx     # b_i
        A[rowV, idx(i,2)] = 1/3*dx^2   # c_i
        A[rowV, idx(i,3)] = 0.25*dx^3  # d_i
        rhs[rowV] = H[i]

        # 2) Value continuity eq : a_i + b_i Δx + c_i Δx^2 + d_i Δx^3 = a_{i+1}
        rowC1 = 4*(i-1) + 2
        A[rowC1, idx(i,0)] = 1.0      # a_i
        A[rowC1, idx(i,1)] = dx       # b_i
        A[rowC1, idx(i,2)] = dx^2     # c_i
        A[rowC1, idx(i,3)] = dx^3     # d_i
        A[rowC1, idx(ip1,0)] = -1.0   # -a_{i+1}
        rhs[rowC1] = 0.0

        # 3) Derivative continuity eq : b_i + 2 c_i Δx + 3 d_i Δx^2 = b_{i+1}
        rowC2 = 4*(i-1) + 3
        A[rowC2, idx(i,1)] = 1.0      # b_i
        A[rowC2, idx(i,2)] = 2.0*dx   # c_i
        A[rowC2, idx(i,3)] = 3.0*dx^2 # d_i
        A[rowC2, idx(ip1,1)] = -1.0   # -b_{i+1}
        rhs[rowC2] = 0.0

        # 4) Second derivative continuity eq : 2 c_i + 6 d_i Δx = 2 c_{i+1}
        rowC3 = 4*(i-1) + 4
        A[rowC3, idx(i,2)] = 2.0      # c_i
        A[rowC3, idx(i,3)] = 6.0*dx   # d_i
        A[rowC3, idx(ip1,2)] = -2.0   # -c_{i+1}
        rhs[rowC3] = 0.0
    end

    # 5) BC : a_1 = a_{n_x}, b_1 = b_{n_x}, c_1 = c_{n_x} (periodicity)
    # Replace the last three equations for boundary conditions
    A[N-2, idx(1,0)] = 1.0
    A[N-2, idx(nx,0)] = -1.0
    rhs[N-2] = 0.0

    A[N-1, idx(1,1)] = 1.0
    A[N-1, idx(nx,1)] = -1.0
    rhs[N-1] = 0.0

    A[N, idx(1,2)] = 1.0
    A[N, idx(nx,2)] = -1.0
    rhs[N] = 0.0

    # Solve for coefficients
    s = A \ rhs

    a, b, c, d = s[1:4:end], s[2:4:end], s[3:4:end], s[4:4:end]

    # Fonction d'interpolation avec extrapolation
    function h_tilde(x_val)
        # Gestion de l'extrapolation
        if extrapolate
            # Si x_val est en dehors du domaine
            if x_val < x[1]
                # Extrapolation cubique à gauche
                xi = x_val - x[1]
                return a[1] + b[1]*xi + c[1]*xi^2 + d[1]*xi^3
            elseif x_val > x[end]
                # Extrapolation cubique à droite
                xi = x_val - x[nx]
                return a[nx] + b[nx]*xi + c[nx]*xi^2 + d[nx]*xi^3
            end
        end
        
        # Trouver dans quelle cellule se trouve x_val (interpolation normale)
        i = 1
        while i < nx && !(x[i] <= x_val && x_val < x[i+1])
            i += 1
        end
        
        # Gestion des cas limites
        if i >= nx
            # Cas spécial pour le dernier point qui peut être exactement égal à x[end]
            if x_val == x[end]
                # Le dernier point appartient à la dernière cellule
                xi = 0.0  # x_end - x_end = 0
                return a[nx]  # Tous les termes avec xi sont nuls
            end
            
            # Point en dehors du domaine à droite
            if !extrapolate
                return 0.0  # Pas d'extrapolation
            else
                # Utiliser la dernière cellule pour l'extrapolation
                xi = x_val - x[nx]
                return a[nx] + b[nx]*xi + c[nx]*xi^2 + d[nx]*xi^3
            end
        elseif i < 1
            # Point en dehors du domaine à gauche
            if !extrapolate
                return 0.0  # Pas d'extrapolation
            else
                # Utiliser la première cellule pour l'extrapolation
                xi = x_val - x[1]
                return a[1] + b[1]*xi + c[1]*xi^2 + d[1]*xi^3
            end
        end
        
        # Interpolation cubique dans la cellule
        xi = x_val - x[i]
        return a[i] + b[i]*xi + c[i]*xi^2 + d[i]*xi^3
    end

    return h_tilde
end