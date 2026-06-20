import streamlit as st
import requests
import numpy as np
from Bio import Align
import streamlit.components.v1 as components

# 1. GLOBAL SYSTEM ARCHITECTURE CONFIGURATION
st.set_page_config(page_title="Macromolecular Analysis Suite v5", layout="wide")

if 'workspace_basket' not in st.session_state:
    st.session_state.workspace_basket = {}

# 2. REST API PIPELINES & PARSERS
def fetch_uniprot_data(gene_query, reviewed_only=True):
    """Queries UniProtKB REST API with automatic param encoding and fallback logic."""
    url = "https://rest.uniprot.org/uniprotkb/search"
    clean_query = gene_query.strip()
    review_status = "true" if reviewed_only else "false"
    
    params = {"query": f"gene:{clean_query} AND reviewed:{review_status}", "format": "json", "size": 8}
    try:
        response = requests.get(url, params=params)
        if response.status_code != 200 or not response.json().get('results', []):
            params["query"] = f"{clean_query} AND reviewed:{review_status}"
            response = requests.get(url, params=params)
            
        if response.status_code == 200:
            parsed_results = []
            for entry in response.json().get('results', []):
                accession = entry.get('primaryAccession', 'N/A')
                org = entry.get('organism', {})
                species_display = org.get('scientificName', 'Unknown Species') + (f" ({org.get('commonName', '')})" if org.get('commonName', '') else "")
                desc = entry.get('proteinDescription', {})
                name_block = desc.get('recommendedName') or desc.get('submissionNames', [{}])[0] or desc.get('alternativeNames', [{}])[0]
                protein_name = name_block.get('fullName', {}).get('value', 'Uncharacterized Protein')
                seq = entry.get('sequence', {}).get('value', '')
                
                parsed_results.append({
                    "id": accession, "name": protein_name, "species": species_display,
                    "length": len(seq), "sequence": seq, "label": f"{protein_name} — [{species_display}] (ID: {accession})"
                })
            return parsed_results
        return []
    except Exception:
        return []

def query_rcsb_pdb(pdb_id):
    """Queries RCSB PDB REST API for structural metadata."""
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id.lower()}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return {
                "success": True, "id": data.get('rcsb_id', pdb_id.upper()),
                "title": data.get('struct', {}).get('title', 'No Title Deposited'),
                "method": data.get('exptl', [{}])[0].get('method', 'Unknown Method'),
                "resolution": f"{data.get('rcsb_entry_info', {}).get('resolution_combined', [None])[0]:.2f} Å" if data.get('rcsb_entry_info', {}).get('resolution_combined', [None])[0] else "N/A",
                "date": data.get('rcsb_accession_info', {}).get('deposit_date', 'Unknown')[:10]
            }
        return {"success": False, "msg": f"PDB Code '{pdb_id}' not found."}
    except Exception as e:
        return {"success": False, "msg": f"Network error: {e}"}

def parse_pdb_ca_coordinates(pdb_id):
    """Downloads raw PDB structure and extracts Alpha Carbon (CA) spatial coordinate matrices."""
    url = f"https://files.rcsb.org/download/{pdb_id.lower()}.pdb"
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return None
        
        coords = []
        for line in response.text.splitlines():
            if line.startswith("ATOM") and line[13:15] == "CA":
                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    coords.append([x, y, z])
                except ValueError:
                    continue
        return np.array(coords) if len(coords) > 0 else None
    except Exception:
        return None

# 3. ADVANCED MATHEMATICAL COMPUTATION MATRICES
def compute_alignment_matrix(seq1, seq2):
    """Calculates pairwise global alignment variables."""
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    alignments = aligner.align(seq1, seq2)
    top = alignments[0]
    lines = format(top).splitlines()
    s1_aligned, s2_aligned = lines[0], lines[2]
    matches = sum(1 for a, b in zip(s1_aligned, s2_aligned) if a == b and a != '-')
    pct = (matches / len(s1_aligned)) * 100 if len(s1_aligned) > 0 else 0
    return pct, top.score, s1_aligned, s2_aligned

def calculate_kabsch_rmsd(coords_P, coords_Q):
    """Implements the Kabsch Algorithm via Singular Value Decomposition (SVD) to find optimal 3D superposition."""
    # Enforce shape uniformity across point sets
    min_len = min(len(coords_P), len(coords_Q))
    P = coords_P[:min_len]
    Q = coords_Q[:min_len]
    
    # Step A: Center coordinates at origin
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    
    # Step B: Compute Covariance Matrix
    covariance_matrix = np.dot(P_centered.T, Q_centered)
    
    # Step C: Singular Value Decomposition (SVD)
    V, S, W_t = np.linalg.svd(covariance_matrix)
    
    # Step D: Enforce right-handed coordinate system (handle reflections)
    d = np.linalg.det(np.dot(W_t.T, V.T))
    if d < 0.0:
        V[:, -1] = -V[:, -1]
        
    # Step E: Calculate Rotation Matrix
    rotation_matrix = np.dot(W_t.T, V.T)
    
    # Step F: Superimpose and compute residual RMSD
    P_rotated = np.dot(P_centered, rotation_matrix)
    diff = P_rotated - Q_centered
    rmsd_score = np.sqrt(np.mean(sum(diff**2)))
    
    return rmsd_score, min_len

# 4. WEBGL CLIENT-SIDE 3D RENDERING COMPONENT
def render_3dmol_viewer(pdb_id, style_mode="cartoon", color_scheme="spectrum"):
    """Embeds an interactive multi-style 3D WebGL structural visualizer canvas without external Python dependencies."""
    html_template = f"""
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <div id="container" style="width: 100%; height: 500px; position: relative; background-color: #111; border-radius: 8px;"></div>
    <script>
        document.addEventListener("DOMContentLoaded", function() {{
            let element = document.getElementById("container");
            let config = {{ backgroundColor: '#111111' }};
            let viewer = $3Dmol.createViewer(element, config);
            
            $3Dmol.download("pdb:{pdb_id.lower()}", viewer, {{}}, function() {{
                viewer.setStyle({{}}, {{ "{style_mode}": {{ color: "{color_scheme}" }} }});
                viewer.zoomTo();
                viewer.render();
            }});
        }});
    </script>
    """
    components.html(html_template, height=520)

def generate_render_html(ref_aligned, tgt_aligned, ref_lbl, tgt_lbl):
    """Renders visual sequence alignment grids."""
    html = "<div style='font-family: monospace; font-size: 13px; white-space: pre; background-color: #141414; color: #fff; padding: 15px; border-radius: 6px; overflow-x: auto; line-height: 1.5;'>"
    l1, l2, l3 = "", "", ""
    chunk = min(100, len(ref_aligned))
    for r, t in zip(ref_aligned[:chunk], tgt_aligned[:chunk]):
        if r == t and r != '-':
            l1 += f"<span style='color: #4caf50;'>{r}</span>"
            l2 += "<span style='color: #4caf50;'>|</span>"
            l3 += f"<span style='color: #4caf50;'>{t}</span>"
        elif r == '-' or t == '-':
            l1 += f"<span style='color: #f44336; background-color: #2b0404;'>{r}</span>"
            l2 += " "
            l3 += f"<span style='color: #f44336; background-color: #2b0404;'>{t}</span>"
        else:
            l1 += f"<span style='color: #2196f3;'>{r}</span>"
            l2 += " "
            l3 += f"<span style='color: #2196f3;'>{t}</span>"
    html += f"<b>REF [{ref_lbl[:12]}]:</b> {l1}<br><b>CONSENSUS MAP   :</b> {l2}<br><b>TGT [{tgt_lbl[:12]}]:</b> {l3}</div>"
    return html

# 5. SIDEBAR NAVIGATION CONTROLLER
with st.sidebar:
    st.header("⚙️ Suite Navigation")
    suite_page = st.selectbox(
        "Choose Analytical Module:",
        ["1. Homology Explorer (Species Focus)", "2. Cross-Protein Workspace (Multi-Gene Focus)", "3. 3D Structural Workstation (PDB Focus)"]
    )
    st.markdown("---")
    st.markdown("### 🧺 Persistent Basket Registry")
    basket_count = len(st.session_state.workspace_basket)
    st.metric(label="Stored Workspace Sequences", value=basket_count)
    if basket_count > 0:
        if st.button("🗑️ Clear Entire Basket", use_container_width=True):
            st.session_state.workspace_basket.clear()
            st.rerun()

# --- MODULE 1: HOMOLOGY EXPLORER ---
if suite_page.startswith("1."):
    st.title("🛡️ Curated Homology Explorer (Single Gene / Multi-Species)")
    st.markdown("Query a single target gene symbol to evaluate sequence retention across curated Swiss-Prot evolution models.")
    gene_in = st.text_input("Enter Gene Symbol (e.g., FUS, SOD1, INS)", value="FUS", key="m1_search")
    if gene_in:
        with st.spinner("Executing parameter-safe UniProt query..."):
            records = fetch_uniprot_data(gene_in, reviewed_only=True)
        if records:
            st.markdown("### 📊 Ortholog Select Matrix")
            selected_items = []
            for idx, item in enumerate(records):
                col_b, col_m = st.columns([0.6, 0.4])
                with col_b:
                    if st.checkbox(f"**{item['species']}** — {item['name']}", value=(idx < 4), key=f"m1_chk_{item['id']}"):
                        selected_items.append(item)
                with col_m:
                    st.caption(f"ID: {item['id']} | Length: {item['length']} aa")
            if len(selected_items) >= 2:
                st.markdown("---")
                ref_species = st.selectbox("Designate Evolutionary Archetype (Reference):", [x['species'] for x in selected_items])
                ref_obj = next(x for x in selected_items if x['species'] == ref_species)
                table_summary = []
                for x in selected_items:
                    if x['id'] == ref_obj['id']:
                        table_summary.append({"Species": f"⭐ {x['species']} (Ref)", "Accession": x['id'], "Identity %": "100.0%", "Score": "Baseline"})
                    else:
                        pct, score, _, _ = compute_alignment_matrix(ref_obj['sequence'], x['sequence'])
                        table_summary.append({"Species": x['species'], "Accession": x['id'], "Identity %": f"{pct:.1f}%", "Score": f"{score:.1f}"})
                st.markdown("### 📈 Computed Evolutionary Identity Values")
                st.table(table_summary)
        else:
            st.warning("No records returned.")

# --- MODULE 2: CROSS-PROTEIN WORKSPACE ---
elif suite_page.startswith("2."):
    st.title("🧺 Cross-Protein Workspace Basket (Multi-Gene Comparative Module)")
    st.markdown("Search for completely different genes sequentially, stack them in your memory basket, and run calculations.")
    col_search, col_basket = st.columns([0.5, 0.5])
    with col_search:
        st.subheader("🔍 Part A: Interrogate & Harvest Sequences")
        workspace_query = st.text_input("Query Any Gene Target:", value="INS")
        if workspace_query:
            with st.spinner("Harvesting records..."):
                ws_hits = fetch_uniprot_data(workspace_query, reviewed_only=True)
            if ws_hits:
                for hit in ws_hits:
                    with st.expander(f"📄 {hit['name']} ({hit['species']})"):
                        st.markdown(f"**Accession:** {hit['id']} | **Size:** {hit['length']} residues")
                        if st.button("📥 Add to Alignment Workspace Basket", key=f"add_ws_{hit['id']}", use_container_width=True):
                            st.session_state.workspace_basket[hit['id']] = hit
                            st.success(f"Registered {hit['id']} to memory!")
                            st.rerun()
            else:
                st.warning("No validated records returned.")
    with col_basket:
        st.subheader("🛒 Part B: Active Workspace Registry")
        if not st.session_state.workspace_basket:
            st.info("Your laboratory workspace is currently empty.")
        else:
            for k, v in list(st.session_state.workspace_basket.items()):
                st.markdown(f"📌 **{v['name']}**\n`Species: {v['species']} | Accession: {k}`")
                if st.button("❌ Remove From List", key=f"rem_{k}"):
                    del st.session_state.workspace_basket[k]
                    st.rerun()
    if len(st.session_state.workspace_basket) >= 2:
        st.markdown("---")
        st.subheader("⚡ Part C: Run Comparative Cross-Protein Matrix")
        b_items = list(st.session_state.workspace_basket.values())
        b_labels = [f"{x['name']} ({x['species']}) [{x['id']}]" for x in b_items]
        ref_label_sel = st.selectbox("Designate Alignment Baseline Reference Object:", b_labels)
        ref_idx = b_labels.index(ref_label_sel)
        ws_ref = b_items[ref_idx]
        if st.button("🚀 Calculate Cross-Protein Alignments", type="primary"):
            st.markdown("### 📊 Structural Multi-Lineage Map View")
            for target_item in b_items:
                if target_item['id'] != ws_ref['id']:
                    pct, score, ref_a, tgt_a = compute_alignment_matrix(ws_ref['sequence'], target_item['sequence'])
                    st.markdown(f"#### 🔍 Comparison: Reference vs **{target_item['name']}** ({target_item['species']})")
                    c1, c2 = st.columns(2)
                    c1.metric("Sequence Identity %", f"{pct:.1f}%")
                    c2.metric("Matrix Alignment Score", f"{score:.1f}")
                    st.markdown(generate_render_html(ref_a, tgt_a, ws_ref['name'], target_item['name']), unsafe_allow_html=True)

# --- MODULE 3: 3D STRUCTURAL WORKSTATION (UPGRADED) ---
elif suite_page.startswith("3."):
    st.title("🏛️ 3D Structural Workstation & Spatial Superposition Engine")
    st.markdown("Transition from 1D character text strings to active tertiary spatial visualization and topological structural overlap analysis.")
    
    tab1, tab2 = st.tabs(["🔍 Interactive 3D Visual Viewer", "📐 Mathematical Spatial Superposition"])
    
    with tab1:
        st.subheader("Interactive WebGL Macromolecular Canvas")
        v_col1, v_col2 = st.columns([0.3, 0.7])
        with v_col1:
            pdb_target = st.text_input("Target PDB Code:", value="1FUS").strip()
            style_mode = st.selectbox("Rendering Geometry:", ["cartoon", "sphere", "stick", "line"])
            color_scheme = st.selectbox("Color Mapping Vector:", ["spectrum", "chain", "ssSecondary", "white", "blue"])
            
            if pdb_target:
                meta = query_rcsb_pdb(pdb_target)
                if meta['success']:
                    st.markdown("---")
                    st.markdown(f"**Method:** {meta['method']}")
                    st.markdown(f"**Resolution:** {meta['resolution']}")
                    st.caption(f"Deposited: {meta['date']}")
                    st.info(f"Title: {meta['title']}")
        with v_col2:
            if pdb_target:
                with st.spinner("Generating WebGL canvas environment..."):
                    render_3dmol_viewer(pdb_target, style_mode, color_scheme)
                    
    with tab2:
        st.subheader("Deterministic Kabsch Backbone Superposition")
        st.markdown("Input two distinct structural models to calculate the spatial divergence of their carbon-alpha backbones.")
        
        sup_col1, sup_col2 = st.columns(2)
        with sup_col1:
            ref_pdb = st.text_input("Reference PDB Structure Archetype:", value="1FUS").strip()
        with sup_col2:
            tgt_pdb = st.text_input("Target PDB Structure Variant:", value="2A5D").strip()
            
        if st.button("🚀 Execute 3D Superposition Matrix", type="primary"):
            if ref_pdb and tgt_pdb:
                with st.spinner("Downloading coordinates and computing SVD matrices..."):
                    coords_ref = parse_pdb_ca_coordinates(ref_pdb)
                    coords_tgt = parse_pdb_ca_coordinates(tgt_pdb)
                    
                if coords_ref is not None and coords_tgt is not None:
                    rmsd, residues_aligned = calculate_kabsch_rmsd(coords_ref, coords_tgt)
                    
                    st.markdown("---")
                    st.success("Mathematical Superposition Complete!")
                    
                    m_col1, m_col2 = st.columns(2)
                    m_col1.metric(label="Calculated Backbone RMSD Score", value=f"{rmsd:.4f} Å")
                    m_col2.metric(label="Paired Structural Carbon-Alpha Nodes", value=f"{residues_aligned} Res")
                    
                    # Evaluation analysis
                    if rmsd < 2.0:
                        st.balloons()
                        st.info("🧬 **Interpretation:** Highly conserved tertiary folding architecture. The structural topology matches downstream lineage specifications.")
                    else:
                        st.warning("⚠️ **Interpretation:** Structural domain divergence detected. Functional transformations or structural drift are present between these conformations.")
                        
                    st.latex(r"\text{RMSD} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} d_i^2}")
                else:
                    st.error("Coordinate extraction failed. Ensure both codes represent valid PDB depositions containing text ATOM lines.")
