import json
import streamlit.components.v1 as components

def render_interactive_map(history, arena_dim, step_idx, mode, selected_drone_id):
    """
    Renders an advanced HTML5 Canvas visualization of the drone swarm with tactical FX.
    """
    
    clean_history = []
    for frame in history:
        clean_frame = {
            "drones": [{"id": d.drone_id, "x": float(d.position[0]), "y": float(d.position[1]), "battery": float(d.battery)} for d in frame["drones"]],
            "edges": [{"source": e["source"], "target": e["target"], "weight": float(e["weight"])} for e in frame.get("edges", [])],
            "obstacles": [{"x": float(o["x"]), "y": float(o["y"]), "r": float(o["radius"]), "dynamic": bool(o["is_dynamic"])} for o in frame.get("obstacles", [])],
            "targets": [{"x": float(t["x"]), "y": float(t["y"]), "reached": bool(t["is_reached"]), "id": int(t["target_id"])} for t in frame.get("targets", [])]
        }
        clean_history.append(clean_frame)

    data_json = json.dumps({
        "history": clean_history,
        "arenaDim": arena_dim,
        "stepIdx": step_idx,
        "mode": mode,
        "selectedDroneId": selected_drone_id
    })

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background-color: #0F172A; /* Dark tactical background */
                display: flex;
                justify-content: center;
                align-items: center;
                overflow: hidden;
                font-family: -apple-system, sans-serif;
                color: #F8FAFC;
            }}
            .canvas-container {{
                position: relative;
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 0 40px rgba(37, 99, 235, 0.2);
                border: 1px solid #1E293B;
            }}
            canvas {{
                background-color: #020617;
                display: block;
            }}
            /* CRT Scanline Overlay */
            .scanlines {{
                position: absolute;
                top: 0; left: 0; width: 100%; height: 100%;
                background: linear-gradient(rgba(18, 16, 16, 0) 50%, rgba(0, 0, 0, 0.25) 50%), linear-gradient(90deg, rgba(255, 0, 0, 0.06), rgba(0, 255, 0, 0.02), rgba(0, 0, 255, 0.06));
                background-size: 100% 4px, 3px 100%;
                pointer-events: none;
                z-index: 10;
                opacity: 0.7;
            }}
            /* Vignette */
            .vignette {{
                position: absolute;
                top: 0; left: 0; width: 100%; height: 100%;
                background: radial-gradient(circle, rgba(0,0,0,0) 50%, rgba(0,0,0,0.8) 100%);
                pointer-events: none;
                z-index: 11;
            }}
            #controls {{
                position: absolute;
                bottom: 20px;
                background: rgba(15, 23, 42, 0.85);
                backdrop-filter: blur(10px);
                border: 1px solid #334155;
                padding: 10px 20px;
                border-radius: 20px;
                box-shadow: 0 4px 20px rgba(0,0,0,0.5);
                display: flex;
                gap: 15px;
                align-items: center;
                z-index: 20;
            }}
            button {{
                background: #3B82F6;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 700;
                transition: all 0.2s;
                text-transform: uppercase;
                letter-spacing: 1px;
                font-size: 11px;
            }}
            button:hover {{ background: #60A5FA; box-shadow: 0 0 10px #3B82F6; }}
            input[type=range] {{ width: 200px; accent-color: #3B82F6; }}
            
            .legend {{
                position: absolute;
                top: 15px;
                right: 15px;
                background: rgba(15, 23, 42, 0.85);
                backdrop-filter: blur(10px);
                padding: 12px;
                border-radius: 8px;
                font-size: 12px;
                border: 1px solid #334155;
                z-index: 20;
            }}
            
            .hud {{
                position: absolute;
                top: 15px;
                left: 15px;
                font-size: 24px;
                font-weight: 900;
                color: #F8FAFC;
                text-shadow: 0 0 10px rgba(255,255,255,0.5);
                letter-spacing: 2px;
                z-index: 20;
            }}
        </style>
    </head>
    <body>
        <div class="canvas-container">
            <div class="hud" id="hud">MISSION VIEW</div>
            <div class="scanlines"></div>
            <div class="vignette"></div>
            <canvas id="arena"></canvas>
            <div class="legend" id="legend">
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
                    <div style="width:12px; height:2px; background:#10B981; box-shadow:0 0 5px #10B981;"></div> Strong Link
                </div>
                <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">
                    <div style="width:12px; height:2px; background:#F59E0B; box-shadow:0 0 5px #F59E0B;"></div> Weak Link
                </div>
                <div style="display:flex; align-items:center; gap:8px;">
                    <div style="width:12px; height:2px; background:#EF4444; border-top:2px dashed #EF4444;"></div> Partition Risk
                </div>
            </div>
            <div id="controls">
                <button id="playBtn">Play ▶</button>
                <input type="range" id="scrubber" min="0" value="0" step="1">
                <span id="stepCounter" style="font-weight:bold; color:#94A3B8; font-variant-numeric: tabular-nums;">STEP 0</span>
            </div>
        </div>
        
        <script>
            const config = {data_json};
            const historyData = config.history;
            const totalSteps = historyData.length;
            const numDrones = historyData[0].drones.length;
            
            let currentStep = config.stepIdx;
            let targetStep = config.stepIdx;
            let playInterval = null;
            let isPlaying = false;
            let interp = 0.0; 
            
            const canvas = document.getElementById("arena");
            const ctx = canvas.getContext("2d");
            
            const size = 680;
            canvas.width = size;
            canvas.height = size;
            const scale = size / config.arenaDim;
            
            // Bright neon colors for dark tactical theme
            const dronePalette = ["#60A5FA", "#34D399", "#A78BFA", "#FBBF24", "#F472B6", "#38BDF8", "#4ADE80"];
            
            let time = 0;
            
            const playBtn = document.getElementById("playBtn");
            const scrubber = document.getElementById("scrubber");
            const stepCounter = document.getElementById("stepCounter");
            
            scrubber.max = totalSteps - 1;
            scrubber.value = currentStep;
            
            const hud = document.getElementById("hud");
            hud.innerText = config.mode.toUpperCase() + " VIEW";
            
            function togglePlay() {{
                isPlaying = !isPlaying;
                playBtn.innerText = isPlaying ? "Pause ⏸" : "Play ▶";
            }}
            
            playBtn.addEventListener("click", togglePlay);
            
            scrubber.addEventListener("input", (e) => {{
                targetStep = parseInt(e.target.value);
                currentStep = targetStep;
                interp = 0;
                stepCounter.innerText = "STEP " + targetStep;
                if (isPlaying) togglePlay();
            }});
            
            function lerp(start, end, amt) {{
                return (1-amt)*start + amt*end;
            }}
            
            function drawGrid() {{
                // Animated pulse grid
                const pulse = Math.sin(time) * 0.1 + 0.15;
                ctx.strokeStyle = `rgba(30, 41, 59, ${{pulse}})`;
                ctx.lineWidth = 1;
                const gridSpacing = 2.0;
                for (let i = 0; i <= config.arenaDim; i += gridSpacing) {{
                    ctx.beginPath(); ctx.moveTo(i * scale, 0); ctx.lineTo(i * scale, size); ctx.stroke();
                    ctx.beginPath(); ctx.moveTo(0, i * scale); ctx.lineTo(size, i * scale); ctx.stroke();
                }}
            }}
            
            function getInterpolatedState(step, nextStep, t) {{
                const f1 = historyData[step];
                const f2 = historyData[nextStep];
                
                const drones = f1.drones.map((d1, i) => {{
                    const d2 = f2.drones[i];
                    return {{
                        id: d1.id,
                        x: lerp(d1.x, d2.x, t),
                        y: lerp(d1.y, d2.y, t),
                        battery: lerp(d1.battery, d2.battery, t)
                    }};
                }});
                
                return {{ drones, edges: f1.edges, targets: f1.targets, obstacles: f1.obstacles }};
            }}
            
            function drawDrone(drone, isSelected) {{
                const color = dronePalette[drone.id % dronePalette.length];
                const x = drone.x * scale;
                const y = drone.y * scale;
                
                ctx.globalCompositeOperation = "lighter";
                
                // Deep glow
                ctx.shadowBlur = isSelected ? 30 : 15;
                ctx.shadowColor = color;
                
                // Base core
                ctx.fillStyle = color;
                ctx.beginPath();
                ctx.arc(x, y, isSelected ? 8 : 6, 0, Math.PI*2);
                ctx.fill();
                
                // Outer rotating prop ring
                ctx.strokeStyle = color;
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.arc(x, y, isSelected ? 16 : 12, time * 3, time * 3 + Math.PI*1.5);
                ctx.stroke();
                
                ctx.shadowBlur = 0;
                ctx.globalCompositeOperation = "source-over";
                
                // Battery Ring (inner)
                const batRatio = drone.battery / 100.0;
                ctx.strokeStyle = batRatio > 0.5 ? "#10B981" : (batRatio > 0.25 ? "#F59E0B" : "#EF4444");
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(x, y, 9, -Math.PI/2, -Math.PI/2 + batRatio * Math.PI * 2);
                ctx.stroke();
                
                // ID Badge
                ctx.fillStyle = "#F8FAFC";
                ctx.font = "bold 11px monospace";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText("D" + drone.id, x, y - 24);
                
                if (config.mode === 'quantum' && isSelected) {{
                    ctx.strokeStyle = "rgba(167, 139, 250, 0.6)";
                    ctx.shadowBlur = 10;
                    ctx.shadowColor = "#A78BFA";
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.arc(x, y, 45 + Math.sin(time*6)*8, 0, Math.PI*2);
                    ctx.stroke();
                    ctx.shadowBlur = 0;
                }}
                
                // Radar sweep for selected drone
                if (isSelected) {{
                    const startAngle = time * 4;
                    const grad = ctx.createRadialGradient(x, y, 0, x, y, 60);
                    grad.addColorStop(0, "rgba(96, 165, 250, 0.4)");
                    grad.addColorStop(1, "rgba(96, 165, 250, 0)");
                    
                    ctx.fillStyle = grad;
                    ctx.beginPath();
                    ctx.moveTo(x, y);
                    ctx.arc(x, y, 60, startAngle, startAngle + Math.PI/3);
                    ctx.closePath();
                    ctx.fill();
                }}
            }}
            
            function drawLink(link, drones) {{
                const d1 = drones.find(d => d.id === link.source);
                const d2 = drones.find(d => d.id === link.target);
                if (!d1 || !d2) return;
                
                let q = link.weight;
                let stroke = "#10B981"; // Emerald
                let w = 2.5;
                if (q < 0.6) {{ stroke = "#F59E0B"; w = 1.5; }}
                if (q < 0.2) {{ stroke = "#EF4444"; w = 1; ctx.setLineDash([4,4]); }}
                
                if (config.mode === 'mission') ctx.globalAlpha = 0.4;
                if (config.mode === 'mesh') ctx.globalAlpha = 1.0;
                if (config.mode === 'coverage') ctx.globalAlpha = 0.15;
                
                ctx.globalCompositeOperation = "lighter";
                ctx.shadowBlur = 10;
                ctx.shadowColor = stroke;
                
                ctx.strokeStyle = stroke;
                ctx.lineWidth = w;
                ctx.beginPath();
                ctx.moveTo(d1.x * scale, d1.y * scale);
                ctx.lineTo(d2.x * scale, d2.y * scale);
                ctx.stroke();
                
                ctx.setLineDash([]);
                ctx.shadowBlur = 0;
                ctx.globalCompositeOperation = "source-over";
                
                // Particles along link
                if (q >= 0.2 && config.mode !== 'coverage') {{
                    const particleT = (time * 2.0 + d1.id * 0.3) % 1.0;
                    const px = lerp(d1.x, d2.x, particleT) * scale;
                    const py = lerp(d1.y, d2.y, particleT) * scale;
                    ctx.fillStyle = "#FFF";
                    ctx.shadowBlur = 8;
                    ctx.shadowColor = "#FFF";
                    ctx.beginPath();
                    ctx.arc(px, py, 2.5, 0, Math.PI*2);
                    ctx.fill();
                    ctx.shadowBlur = 0;
                }}
                ctx.globalAlpha = 1.0;
            }}
            
            function render() {{
                time += 0.03;
                
                if (isPlaying) {{
                    interp += 0.05;
                    if (interp >= 1.0) {{
                        interp = 0;
                        currentStep++;
                        if (currentStep >= totalSteps - 1) {{
                            currentStep = totalSteps - 1;
                            isPlaying = false;
                            playBtn.innerText = "Play ▶";
                        }}
                        scrubber.value = currentStep;
                        stepCounter.innerText = "STEP " + currentStep;
                    }}
                }}
                
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                drawGrid();
                
                const nextStep = Math.min(currentStep + 1, totalSteps - 1);
                const state = getInterpolatedState(currentStep, nextStep, interp);
                
                // Draw Trails (Neon)
                ctx.globalCompositeOperation = "lighter";
                for (let i = 0; i < numDrones; i++) {{
                    ctx.beginPath();
                    const color = dronePalette[i % dronePalette.length];
                    ctx.strokeStyle = color;
                    ctx.lineWidth = 2;
                    ctx.shadowBlur = 10;
                    ctx.shadowColor = color;
                    ctx.globalAlpha = 0.4;
                    for (let s = 0; s <= currentStep; s++) {{
                        const f = historyData[s];
                        if (!f.drones[i]) continue;
                        const pt = f.drones[i];
                        if (s === 0) ctx.moveTo(pt.x * scale, pt.y * scale);
                        else ctx.lineTo(pt.x * scale, pt.y * scale);
                    }}
                    if (interp > 0) ctx.lineTo(state.drones[i].x * scale, state.drones[i].y * scale);
                    ctx.stroke();
                }}
                ctx.globalAlpha = 1.0;
                ctx.shadowBlur = 0;
                ctx.globalCompositeOperation = "source-over";
                
                // Draw Coverage Heatmap
                if (config.mode === 'coverage') {{
                    ctx.fillStyle = "rgba(56, 189, 248, 0.08)";
                    ctx.globalCompositeOperation = "lighter";
                    for (let s = 0; s <= currentStep; s++) {{
                        const f = historyData[s];
                        for (let pt of f.drones) {{
                            ctx.beginPath();
                            ctx.arc(pt.x * scale, pt.y * scale, 2.0 * scale, 0, Math.PI*2);
                            ctx.fill();
                        }}
                    }}
                    ctx.globalCompositeOperation = "source-over";
                }}
                
                // Draw Obstacles (Darkred / Gray)
                for (let o of state.obstacles) {{
                    ctx.fillStyle = o.dynamic ? "rgba(153, 27, 27, 0.4)" : "rgba(51, 65, 85, 0.4)";
                    ctx.strokeStyle = o.dynamic ? "#F87171" : "#94A3B8";
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.arc(o.x * scale, o.y * scale, o.r * scale, 0, Math.PI*2);
                    ctx.fill();
                    ctx.stroke();
                    
                    // Draw hazard striping
                    if (o.dynamic) {{
                        ctx.fillStyle = "rgba(248, 113, 113, 0.8)";
                        ctx.font = "10px sans-serif";
                        ctx.textAlign = "center";
                        ctx.fillText("⚠ HAZARD", o.x * scale, o.y * scale);
                    }}
                }}
                
                // Draw Targets (Pulsing Beacons)
                ctx.globalCompositeOperation = "lighter";
                for (let t of state.targets) {{
                    if (t.reached) {{
                        ctx.shadowBlur = 15;
                        ctx.shadowColor = "#34D399";
                        ctx.fillStyle = "rgba(52, 211, 153, 0.3)";
                        ctx.strokeStyle = "#34D399";
                        ctx.beginPath(); ctx.arc(t.x * scale, t.y * scale, 1.2 * scale, 0, Math.PI*2);
                        ctx.fill(); ctx.stroke();
                    }} else {{
                        const pulse = 1.0 * scale + Math.sin(time * 6) * 4;
                        ctx.shadowBlur = 20;
                        ctx.shadowColor = "#FBBF24";
                        ctx.fillStyle = "rgba(251, 191, 36, 0.3)";
                        ctx.strokeStyle = "#FBBF24";
                        ctx.beginPath(); ctx.arc(t.x * scale, t.y * scale, pulse, 0, Math.PI*2);
                        ctx.fill(); ctx.stroke();
                    }}
                }}
                ctx.globalCompositeOperation = "source-over";
                ctx.shadowBlur = 0;
                
                for (let e of state.edges) drawLink(e, state.drones);
                for (let d of state.drones) drawDrone(d, d.id === config.selectedDroneId);
                
                requestAnimationFrame(render);
            }}
            
            render();
            
        </script>
    </body>
    </html>
    """
    
    components.html(html_code, height=720)
