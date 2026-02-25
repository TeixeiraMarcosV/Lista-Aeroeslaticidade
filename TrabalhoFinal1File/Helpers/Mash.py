# ============================================================
# Célula 1 — Imports
# ============================================================
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# ============================================================
# Célula 2 — Geometria: corda linear (asa trapezoidal)
# ============================================================

def chord_linear(y, b, c_root, c_tip):
    return c_root + (c_tip - c_root) * (y / b)

# ============================================================
# Célula 3 — Gerador: QUADS no interior + (RETÂNGULO + TRIÂNGULO) no bordo inclinado
#
# Ideia:
# - Usamos um passo base h = c_tip / nx_tip
# - Em cada faixa (y_j -> y_{j+1}), preenchemos:
#     1) quads completos até x <= min(c_j, c_{j+1})
#     2) se sobrar um pedacinho (<h) até min(c_j, c_{j+1}), criamos 1 quad "parcial"
#     3) se uma das cordas for maior que a outra, o resto vira 1 triângulo (wedge)
#
# Isso remove a restrição |ΔNx| <= 1 e funciona para qualquer malha (ny, nx_tip).
# ============================================================

def generate_mesh_quads_plus_edge_split(
    b,
    c_root,
    c_tip,
    nx_tip,   # define h = c_tip/nx_tip
    ny        # divisões no span
):
    """
    Gera malha para o domínio 0 <= x <= c(y), 0 <= y <= b, com c(y) linear.

    Para cada faixa (y0->y1), o polígono é:
        (0,y0) -> (c0,y0) -> (c1,y1) -> (0,y1)

    Decomposição:
      - QUADS regulares de largura h até x <= min(c0,c1)
      - 1 QUAD parcial se min(c0,c1) não cair exatamente em múltiplo de h
      - 1 TRIÂNGULO para o "wedge" quando c0 != c1

    Retorna:
      nodes: array (N,3) com (x,y,z)
      quads: lista de [n1,n2,n3,n4]
      tris : lista de [n1,n2,n3]
      meta : dicionário com h, y_nodes
    """
    if b <= 0 or c_root <= 0 or c_tip <= 0:
        raise ValueError("b, c_root, c_tip devem ser > 0.")
    if nx_tip <= 0 or ny <= 0:
        raise ValueError("nx_tip e ny devem ser > 0.")

    h = c_tip / nx_tip
    y_nodes = np.linspace(0.0, b, ny + 1)

    # ---------- nós (por coordenada, para suportar borda inclinada) ----------
    nodes = []
    node_id = {}

    def add_node(x, y, z=0.0):
        # chave arredondada p/ evitar duplicatas por erro numérico
        key = (round(float(x), 12), round(float(y), 12), round(float(z), 12))
        if key in node_id:
            return node_id[key]
        idx = len(nodes)
        nodes.append([x, y, z])
        node_id[key] = idx
        return idx

    # ---------- elementos ----------
    quads = []
    tris = []

    for j in range(ny):
        y0 = y_nodes[j]
        y1 = y_nodes[j + 1]

        c0 = chord_linear(y0, b, c_root, c_tip)
        c1 = chord_linear(y1, b, c_root, c_tip)

        minc = min(c0, c1)
        maxc = max(c0, c1)

        # número de quads completos (largura h) que cabem em minc
        n_full = int(np.floor(minc / h + 1e-12))
        x_full_end = n_full * h

        # 1) quads completos
        for i in range(n_full):
            xa = i * h
            xb = (i + 1) * h
            n1 = add_node(xa, y0)
            n2 = add_node(xb, y0)
            n3 = add_node(xb, y1)
            n4 = add_node(xa, y1)
            quads.append([n1, n2, n3, n4])

        # 2) quad parcial até minc (se necessário)
        if (minc - x_full_end) > 1e-10:
            xa = x_full_end
            xb = minc
            n1 = add_node(xa, y0)
            n2 = add_node(xb, y0)
            n3 = add_node(xb, y1)
            n4 = add_node(xa, y1)
            quads.append([n1, n2, n3, n4])

        # 3) triângulo do wedge (se existir)
        if (maxc - minc) > 1e-10:
            if c0 > c1:
                # corda diminui com y: triângulo fica "embaixo" (linha y0 é maior)
                # vértices: (minc,y0) -> (c0,y0) -> (minc,y1)  (onde minc=c1)
                t = [add_node(minc, y0), add_node(c0, y0), add_node(minc, y1)]
                tris.append(t)
            else:
                # corda aumenta com y: triângulo fica "em cima" (linha y1 é maior)
                # vértices: (minc,y0) -> (minc,y1) -> (c1,y1)  (onde minc=c0)
                t = [add_node(minc, y0), add_node(minc, y1), add_node(c1, y1)]
                tris.append(t)

    nodes = np.array(nodes, dtype=float)
    meta = {"h": h, "y_nodes": y_nodes}
    return nodes, quads, tris, meta

# ============================================================
# Célula 4 — Plot 2D (pra validar se ficou igual ao seu desenho)
# ============================================================

def plot_mesh_2d(nodes, quads, tris, title="Malha 2D (eixos invertidos)", path=None):
    fig = plt.figure(figsize=(9,4))

    # quads
    for q in quads:
        P = nodes[q][:, :2]
        P = np.vstack([P, P[0]])
        plt.plot(P[:,1], P[:,0], linewidth=1)

    # tris
    for t in tris:
        P = nodes[t][:, :2]
        P = np.vstack([P, P[0]])
        plt.plot(P[:,1], P[:,0], linewidth=1)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("y (envergadura)")
    plt.ylabel("x (corda)")
    plt.title(title)
    plt.tight_layout()

    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
# ============================================================
# Célula 5 — Plot 3D (z=0)
# ============================================================

def plot_mesh_3d(nodes, quads, tris, title="Malha 3D (eixos invertidos)", path=None):
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection="3d")

    faces = []

    for q in quads:
        P = nodes[q]
        faces.append(np.column_stack([P[:,1], P[:,0], P[:,2]]))

    for t in tris:
        P = nodes[t]
        faces.append(np.column_stack([P[:,1], P[:,0], P[:,2]]))

    poly = Poly3DCollection(faces, edgecolor="k", linewidths=0.4, alpha=0.95)
    ax.add_collection3d(poly)

    xs = nodes[:,1]
    ys = nodes[:,0]
    zs = nodes[:,2]

    ax.set_xlim(xs.min(), xs.max())
    ax.set_ylim(ys.min(), ys.max())
    ax.set_zlim(-0.05*(ys.max()-ys.min()+1e-9),
                 0.05*(ys.max()-ys.min()+1e-9))

    ax.set_xlabel("y (envergadura)")
    ax.set_ylabel("x (corda)")
    ax.set_zlabel("z")
    ax.set_title(title)

    ax.view_init(elev=20, azim=-60)
    plt.tight_layout()

    if path is not None:
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# ============================================================
# Célula 7 — Matrizes de rigidez (K) e massa (M) + modos de vibrar
# (FEM de placa Mindlin-Reissner com triângulos lineares)
# ============================================================
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import lil_matrix, csc_matrix
from scipy.sparse.linalg import eigsh

def quads_to_tris(quads):
    """Divide cada quad [n1,n2,n3,n4] em 2 triângulos (n1,n2,n3) e (n1,n3,n4)."""
    tris_from_quads = []
    for (n1,n2,n3,n4) in quads:
        tris_from_quads.append([n1,n2,n3])
        tris_from_quads.append([n1,n3,n4])
    return np.array(tris_from_quads, dtype=int)

def mindlin_tri3_element_matrices(xy, E, nu, h, rho, kappa=5/6):
    """
    Elemento TRI3 Mindlin com DOFs por nó: [w, tx, ty]
    Retorna Ke (9x9) e Me (9x9) (massa consistente).
    """
    x1,y1 = xy[0]
    x2,y2 = xy[1]
    x3,y3 = xy[2]

    # Área (duas vezes)
    detJ2 = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
    A = 0.5*detJ2
    if A <= 0:
        raise ValueError("Triângulo com área não-positiva (ordem dos nós incorreta?)")

    # Derivadas das funções de forma lineares (constantes no elemento)
    # N_i = (a_i + b_i x + c_i y) / (2A)
    b1, c1 = (y2 - y3), (x3 - x2)
    b2, c2 = (y3 - y1), (x1 - x3)
    b3, c3 = (y1 - y2), (x2 - x1)

    dNdx = np.array([b1, b2, b3], dtype=float) / (2*A)
    dNdy = np.array([c1, c2, c3], dtype=float) / (2*A)

    # ---------- Bending ----------
    D = E*h**3/(12*(1-nu**2))
    Db = D*np.array([[1,   nu, 0],
                     [nu,  1,  0],
                     [0,   0, (1-nu)/2]], dtype=float)

    Bb = np.zeros((3, 9), dtype=float)
    for i in range(3):
        wi  = 3*i
        txi = 3*i + 1
        tyi = 3*i + 2
        # kx = dtx/dx
        Bb[0, txi] = dNdx[i]
        # ky = dty/dy
        Bb[1, tyi] = dNdy[i]
        # kxy = dtx/dy + dty/dx
        Bb[2, txi] = dNdy[i]
        Bb[2, tyi] = dNdx[i]

    Kb = (Bb.T @ Db @ Bb) * A

    # ---------- Shear (1 ponto no centróide para reduzir locking) ----------
    G = E/(2*(1+nu))
    Ds = (kappa*G*h) * np.eye(2)

    Bs = np.zeros((2, 9), dtype=float)
    Ncent = 1/3  # N1=N2=N3 no centróide
    for i in range(3):
        wi  = 3*i
        txi = 3*i + 1
        tyi = 3*i + 2

        # gx = tx + dw/dx
        Bs[0, wi]  = dNdx[i]
        Bs[0, txi] = Ncent

        # gy = ty + dw/dy
        Bs[1, wi]  = dNdy[i]
        Bs[1, tyi] = Ncent

    Ks = (Bs.T @ Ds @ Bs) * A

    Ke = Kb + Ks

    # ---------- Massa consistente ----------
    # Matriz consistente para triângulo linear: (A/12)*[[2,1,1],[1,2,1],[1,1,2]]
    Mtri = (A/12) * np.array([[2,1,1],[1,2,1],[1,1,2]], dtype=float)

    Me = np.zeros((9, 9), dtype=float)

    # massa translacional (w)
    mw = rho*h
    for i in range(3):
        for j in range(3):
            Me[3*i, 3*j] = mw * Mtri[i, j]

    # inércia rotacional (tx, ty)
    mr = rho*(h**3)/12
    for i in range(3):
        for j in range(3):
            Me[3*i+1, 3*j+1] = mr * Mtri[i, j]
            Me[3*i+2, 3*j+2] = mr * Mtri[i, j]

    return Ke, Me

def assemble_KM_mindlin(nodes, tris, E, nu, h, rho, kappa=5/6):
    """
    Monta K e M globais (sparse) para placa Mindlin TRI3.
    """
    n_nodes = nodes.shape[0]
    ndof = 3*n_nodes
    K = lil_matrix((ndof, ndof), dtype=float)
    M = lil_matrix((ndof, ndof), dtype=float)

    for tri in tris:
        xy = nodes[np.array(tri), :2]
        Ke, Me = mindlin_tri3_element_matrices(xy, E=E, nu=nu, h=h, rho=rho, kappa=kappa)

        dofs = []
        for nid in tri:
            dofs.extend([3*nid, 3*nid+1, 3*nid+2])
        dofs = np.array(dofs, dtype=int)

        for a in range(9):
            ia = dofs[a]
            for b in range(9):
                ib = dofs[b]
                K[ia, ib] += Ke[a, b]
                M[ia, ib] += Me[a, b]

    return K.tocsc(), M.tocsc()

def apply_clamped_root_bc(nodes, K, M, y_root=0.0, tol=1e-12):
    """
    Engaste na raiz: y == y_root => w=tx=ty=0.
    Retorna Kff, Mff e o mapeamento de DOFs livres.
    """
    y = nodes[:,1]
    root_nodes = np.where(np.isclose(y, y_root, atol=tol))[0]
    fixed = []
    for nid in root_nodes:
        fixed.extend([3*nid, 3*nid+1, 3*nid+2])
    fixed = np.array(sorted(set(fixed)), dtype=int)

    ndof = K.shape[0]
    all_dofs = np.arange(ndof, dtype=int)
    free = np.setdiff1d(all_dofs, fixed)

    Kff = K[free][:, free]
    Mff = M[free][:, free]
    return Kff, Mff, free, fixed, root_nodes

def solve_modes(Kff, Mff, n_modes=6, sigma=0.0):
    """
    Resolve K φ = ω² M φ com eigsh (menores ω).
    Retorna (omega, modes) ordenados.
    """
    # shift-invert ajuda a pegar os menores autovalores com mais robustez
    vals, vecs = eigsh(Kff, k=n_modes, M=Mff, sigma=sigma, which='LM')
    vals = np.real(vals)
    vecs = np.real(vecs)

    # ω² = λ
    idx = np.argsort(vals)
    vals = vals[idx]
    vecs = vecs[:, idx]

    omega = np.sqrt(np.clip(vals, 0, None))
    return omega, vecs

def plot_mode_w(nodes, tris, mode_w, title):
    """Plota o deslocamento transversal w em uma malha triangular."""
    import matplotlib.tri as mtri
    tri_obj = mtri.Triangulation(nodes[:,0], nodes[:,1], tris)
    plt.figure()
    plt.tripcolor(tri_obj, mode_w, shading='gouraud')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.colorbar(label='w (modo arbitrário)')
    plt.title(title)
    plt.xlabel('x'); plt.ylabel('y')
    plt.show()
