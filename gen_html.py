import json

with open("output/graph_data_embedded.json", "r", encoding="utf-8") as f:
    data = json.load(f)

json_str = json.dumps(data, ensure_ascii=False)

html = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>Diffusion Model for Anomaly Detection - Citation Graph</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #1a1a2e; color: #eee; overflow: hidden; }
        #header { padding: 12px 20px; background: linear-gradient(135deg, #16213e, #0f3460); border-bottom: 1px solid #333; display: flex; align-items: center; gap: 20px; }
        #header h1 { font-size: 1.2rem; color: #00d4ff; }
        #header .stats { font-size: 0.8rem; color: #888; }
        #controls { padding: 8px 20px; background: #16213e; border-bottom: 1px solid #333; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
        .ctrl-btn { background: #0f3460; color: #00d4ff; border: 1px solid #00d4ff; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 0.75rem; }
        .ctrl-btn:hover, .ctrl-btn.active { background: #00d4ff; color: #1a1a2e; }
        .ctrl-input { background: #1a1a2e; border: 1px solid #333; color: #fff; padding: 4px 10px; border-radius: 4px; font-size: 0.8rem; width: 250px; }
        #container { display: flex; height: calc(100vh - 88px); }
        #graph { flex: 1; background: #1a1a2e; }
        #sidebar { width: 370px; background: #16213e; border-left: 1px solid #333; overflow-y: auto; padding: 12px; }
        #sidebar h3 { color: #00d4ff; margin-bottom: 8px; font-size: 0.85rem; border-bottom: 1px solid #333; padding-bottom: 4px; }
        .pcard { background: #1a1a2e; border-radius: 6px; padding: 8px; margin-bottom: 6px; border: 1px solid #333; cursor: pointer; transition: border-color 0.2s; }
        .pcard:hover { border-color: #00d4ff; }
        .pcard .pt { font-size: 0.78rem; color: #fff; margin-bottom: 3px; line-height: 1.3; }
        .pcard .pm { font-size: 0.68rem; color: #888; display: flex; gap: 6px; flex-wrap: wrap; align-items: center; }
        .tag { display: inline-block; padding: 1px 5px; border-radius: 3px; font-size: 0.62rem; }
        .tm { background: #1e3a5f; color: #60a5fa; }
        .td { background: #1e3a2f; color: #6bcf7f; }
        .ty { background: #3a1e3f; color: #c084fc; }
        .legend { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0; }
        .legend-item { display: flex; align-items: center; gap: 4px; font-size: 0.7rem; }
        .ldot { width: 10px; height: 10px; border-radius: 50%; }
        #dp { display: none; margin-bottom: 15px; }
        #dp .cb { float: right; cursor: pointer; color: #888; font-size: 1.1rem; }
        #dp .cb:hover { color: #fff; }
        .dt { font-size: 0.85rem; color: #fff; margin: 8px 0; line-height: 1.4; }
        .dr { font-size: 0.72rem; color: #aaa; margin: 3px 0; }
        .dr strong { color: #00d4ff; }
    </style>
</head>
<body>
    <div id="header">
        <h1>Diffusion Model x Anomaly Detection - Citation Network</h1>
        <span class="stats" id="stats"></span>
    </div>
    <div id="controls">
        <input type="text" class="ctrl-input" id="search" placeholder="Search papers by title..." />
        <button class="ctrl-btn active" id="btn-all" onclick="filterMethod(null)">All</button>
        <button class="ctrl-btn" id="btn-recon" onclick="filterMethod('reconstruction_based')">Reconstruction</button>
        <button class="ctrl-btn" id="btn-score" onclick="filterMethod('score_based')">Score-based</button>
        <button class="ctrl-btn" id="btn-hybrid" onclick="filterMethod('hybrid')">Hybrid</button>
        <button class="ctrl-btn" id="btn-like" onclick="filterMethod('likelihood_based')">Likelihood</button>
        <button class="ctrl-btn" onclick="resetView()">Reset View</button>
    </div>
    <div id="container">
        <div id="graph"></div>
        <div id="sidebar">
            <div id="dp">
                <span class="cb" onclick="closeDetail()">&times;</span>
                <h3>Paper Details</h3>
                <div id="dc"></div>
            </div>
            <h3>Legend</h3>
            <div class="legend">
                <div class="legend-item"><div class="ldot" style="background:#00d4ff"></div>2026</div>
                <div class="legend-item"><div class="ldot" style="background:#ff6b6b"></div>2025</div>
                <div class="legend-item"><div class="ldot" style="background:#ffd93d"></div>2024</div>
                <div class="legend-item"><div class="ldot" style="background:#6bcf7f"></div>2023</div>
                <div class="legend-item"><div class="ldot" style="background:#a855f7"></div>Earlier</div>
            </div>
            <div class="legend">
                <div class="legend-item"><div class="ldot" style="background:#fff;width:18px;height:18px"></div>High relevance</div>
                <div class="legend-item"><div class="ldot" style="background:#fff;width:8px;height:8px"></div>Low relevance</div>
            </div>
            <h3 style="margin-top:12px">Core Papers (Relevance >= 8)</h3>
            <div id="core"></div>
        </div>
    </div>
    <script>
    var GD = __JSON_PLACEHOLDER__;
    var network, nds, eds;
    var yc = {2026:'#00d4ff',2025:'#ff6b6b',2024:'#ffd93d',2023:'#6bcf7f',2022:'#f97316',2021:'#ec4899'};
    function gc(y){return yc[y]||'#a855f7';}
    function init(){
        var dm={};
        GD.edges.forEach(function(e){dm[e.source]=(dm[e.source]||0)+1;dm[e.target]=(dm[e.target]||0)+1;});
        nds=new vis.DataSet(GD.nodes.map(function(n){
            var deg=dm[n.id]||0;
            var sz=8+(n.relevance||3)*2+deg*3;
            return{id:n.id,label:n.title.length>35?n.title.substring(0,35)+'...':n.title,
                title:'<b>'+n.title+'</b><br>Year: '+(n.year||'N/A')+'<br>Method: '+(n.method||'N/A')+'<br>Domain: '+(n.domain||'N/A'),
                color:{background:gc(n.year),border:gc(n.year),highlight:{background:'#fff',border:gc(n.year)}},
                font:{color:'#ccc',size:9},shape:'dot',size:sz,_d:n,_deg:deg};
        }));
        eds=new vis.DataSet(GD.edges.map(function(e,i){
            return{id:i,from:e.source,to:e.target,arrows:{to:{scaleFactor:0.5}},color:{color:'#333',highlight:'#00d4ff'},width:1};
        }));
        document.getElementById('stats').textContent='Nodes: '+GD.nodes.length+' | Edges: '+GD.edges.length+' | Analyzed: '+GD.nodes.filter(function(n){return n.method;}).length;
        var c=document.getElementById('graph');
        network=new vis.Network(c,{nodes:nds,edges:eds},{
            physics:{enabled:true,solver:'forceAtlas2Based',forceAtlas2Based:{gravitationalConstant:-30,springLength:120,springConstant:0.03,avoidOverlap:0.5},stabilization:{iterations:200,fit:true},maxVelocity:30},
            interaction:{hover:true,tooltipDelay:100,keyboard:true,zoomView:true,dragView:true},
            edges:{smooth:{type:'continuous',roundness:0.2}},nodes:{borderWidth:2}
        });
        network.on('click',function(p){if(p.nodes.length>0){var nd=nds.get(p.nodes[0]);showDetail(nd);}});
        network.on('doubleClick',function(p){if(p.nodes.length>0){network.focus(p.nodes[0],{scale:2,animation:{duration:500,easingFunction:'easeInOutQuad'}});}});
        renderCore();
    }
    function showDetail(nd){
        var d=nd._d;
        document.getElementById('dp').style.display='block';
        var h='<div class="dt">'+d.title+'</div>';
        h+='<div class="dr"><strong>Year:</strong> '+(d.year||'N/A')+'</div>';
        h+='<div class="dr"><strong>Method:</strong> '+(d.method||'N/A')+'</div>';
        h+='<div class="dr"><strong>Domain:</strong> '+(d.domain||'N/A')+'</div>';
        h+='<div class="dr"><strong>Relevance:</strong> '+(d.relevance||'N/A')+'/10</div>';
        h+='<div class="dr"><strong>Connections:</strong> '+(nd._deg||0)+'</div>';
        if(d.contribution)h+='<div class="dr" style="margin-top:6px"><strong>Contribution:</strong><br>'+d.contribution+'</div>';
        document.getElementById('dc').innerHTML=h;
    }
    function closeDetail(){document.getElementById('dp').style.display='none';}
    function filterMethod(m){
        document.querySelectorAll('.ctrl-btn').forEach(function(b){b.classList.remove('active');});
        if(m===null){
            document.getElementById('btn-all').classList.add('active');
            nds.forEach(function(n){nds.update({id:n.id,hidden:false});});
        }else{
            var bid={'reconstruction_based':'btn-recon','score_based':'btn-score','hybrid':'btn-hybrid','likelihood_based':'btn-like'};
            var el=document.getElementById(bid[m]);if(el)el.classList.add('active');
            nds.forEach(function(n){nds.update({id:n.id,hidden:!(!n._d.method||n._d.method===m)});});
        }
    }
    function resetView(){filterMethod(null);network.fit({animation:{duration:500,easingFunction:'easeInOutQuad'}});}
    function renderCore(){
        var core=GD.nodes.filter(function(n){return n.relevance>=8;}).sort(function(a,b){return b.relevance-a.relevance;}).slice(0,15);
        var el=document.getElementById('core');
        el.innerHTML=core.map(function(n){
            var h='<div class="pcard" onclick="focusNode(\''+n.id+'\')">';
            h+='<div class="pt">'+n.title+'</div><div class="pm">';
            h+='<span class="tag ty">'+(n.year||'N/A')+'</span>';
            if(n.method)h+='<span class="tag tm">'+n.method.replace(/_/g,' ')+'</span>';
            if(n.domain)h+='<span class="tag td">'+n.domain.replace(/_/g,' ')+'</span>';
            h+='<span style="color:#ffd93d">R:'+n.relevance+'</span></div></div>';
            return h;
        }).join('');
    }
    function focusNode(id){
        network.selectNodes([id]);
        network.focus(id,{scale:1.5,animation:{duration:500,easingFunction:'easeInOutQuad'}});
        var nd=nds.get(id);if(nd)showDetail(nd);
    }
    document.getElementById('search').addEventListener('input',function(e){
        var q=e.target.value.toLowerCase();
        if(q.length<2)return;
        var ms=GD.nodes.filter(function(n){return n.title.toLowerCase().indexOf(q)>=0;});
        if(ms.length>0&&ms.length<20){
            network.selectNodes(ms.map(function(m){return m.id;}));
            if(ms.length===1)network.focus(ms[0].id,{scale:1.5,animation:true});
        }
    });
    init();
    </script>
</body>
</html>"""

html = html.replace("__JSON_PLACEHOLDER__", json_str)

with open("output/citation_graph.html", "w", encoding="utf-8") as f:
    f.write(html)

print(f"Done. HTML size: {len(html)} bytes")
