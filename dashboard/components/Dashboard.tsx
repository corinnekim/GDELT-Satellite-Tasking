'use client';

import { useEffect, useState, useCallback } from 'react';
import dynamic from 'next/dynamic';
import Split from 'react-split';
import {
  ComposedChart, Line, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid,
} from 'recharts';
import type { Target, SatPass } from './LeafletMap';

const LeafletMap = dynamic(() => import('./LeafletMap'), { ssr: false });

const dashboardPath = (date: string) => `/data/dashboard/daily_${date}.json`;

interface DashboardData {
  date: string;
  generated_at: string;
  summary: { satellite_targets: number; total_passes: number; };
  targets: Target[];
}
interface FlatPass extends SatPass {
  city: string;
  innov_z: number;
  tier: string;
  roiRank?: number;
}
interface TrendPoint {
  date: string;
  z: number;
  goldstein: number;
}
interface History30d {
  date: string;
  z: number;
  sources: number;
  goldstein: number;
}

const fmtUTC = (iso: string) => {
  const d = new Date(iso);
  return `${String(d.getUTCHours()).padStart(2,'0')}:${String(d.getUTCMinutes()).padStart(2,'0')} UTC`;
};
const fmtMD = (iso: string) => {
  const d = new Date(iso);
  return `${d.getUTCMonth()+1}/${d.getUTCDate()}`;
};
const MONTHS = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'];
const fmtDateLabel = (date: string) => {
  const m = parseInt(date.slice(4,6)) - 1;
  return `${date.slice(6,8)} ${MONTHS[m]} ${date.slice(0,4)}`;
};
const fmtDateShort = (date: string) =>
  `${date.slice(4,6)}/${date.slice(6,8)}`;

const riskDisplay = (label: string) => {
  if (label === '위기') return '심각';
  if (label === '위험') return '주의';
  return label; // 심각/주의/관심 그대로
};

const riskClass = (label: string) => {
  if (label === '위기' || label === '심각') return 'red';
  if (label === '위험' || label === '주의') return 'amber';
  return 'yellow'; // 관심
};

function flattenPasses(targets: Target[]): FlatPass[] {
  const sorted = [...targets].sort((a,b) => b.innov_z - a.innov_z);
  const rankMap = new Map(sorted.map((t,i) => [t.city, i+1]));
  return targets.flatMap(t =>
    t.satellite_passes.map(p => ({
      ...p, city: t.city, innov_z: t.innov_z, tier: t.tier,
      roiRank: rankMap.get(t.city),
    }))
  ).sort((a, b) => new Date(a.pass_time_utc).getTime() - new Date(b.pass_time_utc).getTime());
}

function groupPassesByTarget(targets: Target[]): { target: Target; passes: FlatPass[]; rank: number }[] {
  return targets
    .slice()
    .sort((a, b) => b.innov_z - a.innov_z)
    .map((t, idx) => ({
      target: t,
      rank: idx + 1,
      passes: t.satellite_passes
        .map(p => ({ ...p, city: t.city, innov_z: t.innov_z, tier: t.tier, roiRank: idx+1 }))
        .sort((a, b) => new Date(a.pass_time_utc).getTime() - new Date(b.pass_time_utc).getTime()),
    }))
    .filter(g => g.passes.length > 0);
}

const passKey = (p: FlatPass) => `${p.city}-${p.satellite}-${p.pass_time_utc}`;

const S = {
  bg:      '#0a0a0a', bg2: '#111111', bg3: '#1a1a1a', bg4: '#222222',
  border:  'rgba(255,255,255,0.08)', borderB: 'rgba(255,255,255,0.15)',
  red:     '#e05252', redDim:   'rgba(224,82,82,0.12)',
  amber:   '#d4883a', amberDim: 'rgba(212,136,58,0.12)',
  blue:    '#4a90d4', blueDim:  'rgba(74,144,212,0.12)',
  green:   '#5fe6a0', greenDim: 'rgba(95,230,160,0.12)',
  text:    '#ffffff', textSub: '#aaaaaa', textDim: '#555555',
  mono:    "'Martian Mono', monospace",
  body:    "'Plus Jakarta Sans', sans-serif",
  display: "'Plus Jakarta Sans', sans-serif",
} as const;

const globalStyles = `
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background:${S.bg}; color:${S.text}; font-family:${S.body}; min-height:100vh; font-size:13px; -webkit-font-smoothing:antialiased; overflow:hidden; }

  .split-horiz { display:flex; height:100%; }
  .split-horiz > div { min-height:0; }
  .gutter-horizontal {
    cursor:col-resize; background:${S.bg};
    width:5px !important; flex-shrink:0; position:relative; transition:background .15s;
  }
  .gutter-horizontal::after {
    content:''; position:absolute; top:50%; left:50%;
    transform:translate(-50%,-50%);
    width:1px; height:40px; background:rgba(255,255,255,0.15); border-radius:1px;
  }
  .gutter-horizontal:hover { background:rgba(74,144,212,0.2); }
  .gutter-horizontal:hover::after { background:${S.blue}; }
  .gutter-vertical {
    cursor:row-resize; background:${S.bg};
    height:5px !important; flex-shrink:0; position:relative; transition:background .15s;
  }
  .gutter-vertical:hover { background:rgba(74,144,212,0.2); }

  .panel { background:${S.bg2}; border:1px solid ${S.border}; display:flex; flex-direction:column; overflow:hidden; }
  .panel-hdr { display:flex; align-items:center; justify-content:space-between; padding:8px 12px; border-bottom:1px solid ${S.border}; background:${S.bg3}; flex-shrink:0; }
  .panel-title { font-family:${S.body}; font-size:11px; font-weight:600; letter-spacing:.02em; color:${S.textSub}; }
  .panel-tag   { font-family:${S.mono}; font-size:9px; color:${S.textDim}; }

  .badge { display:inline-block; padding:1px 5px; font-size:9px; font-weight:500; border-radius:2px; font-family:${S.body}; }
  .b-red    { background:${S.redDim};   color:${S.red};   border:1px solid rgba(224,82,82,0.2); }
  .b-amber  { background:${S.amberDim}; color:${S.amber}; border:1px solid rgba(212,136,58,0.2); }
  .b-yellow { background:rgba(245,200,66,0.12); color:#f5c842; border:1px solid rgba(245,200,66,0.2); }

  .sch-row { border-left:2px solid transparent; transition:background .1s; }
  .sch-row:hover { background:rgba(255,255,255,0.025); }
  .sch-row.urg { border-left-color:${S.red}; }

  .tl-default  { background:#3a3a3a; color:#aaa; opacity:.8; }
  .tl-approved { background:${S.red}; color:#fff; opacity:.95; }

  .date-sel { appearance:none; background:${S.bg3}; border:1px solid ${S.border}; color:${S.text}; font-family:${S.body}; font-size:11px; padding:4px 22px 4px 8px; cursor:pointer; outline:none; }
  .date-sel:hover { border-color:${S.borderB}; }

  .approve-btn {
    width:100%; padding:8px; cursor:pointer; font-family:${S.body};
    font-size:11px; font-weight:500; letter-spacing:.04em; transition:all .15s;
    background:${S.redDim}; border:1px solid rgba(224,82,82,0.3); color:${S.red};
  }
  .approve-btn:hover { background:rgba(224,82,82,0.2); }
  .approve-btn.done { background:${S.greenDim}; border-color:rgba(95,230,160,0.3); color:${S.green}; }

  .art-link { display:flex; gap:8px; padding:5px 8px; background:${S.bg3}; border:1px solid ${S.border}; text-decoration:none; transition:border-color .15s; margin-bottom:3px; }
  .art-link:hover { border-color:${S.borderB}; }
  .ci-fill { height:100%; background:${S.red}; opacity:.6; }

  .cart-item { display:flex; align-items:center; gap:8px; padding:5px 10px; border-bottom:1px solid rgba(255,255,255,0.04); }
  .cart-item:last-child { border-bottom:none; }

  ::-webkit-scrollbar { width:3px; }
  ::-webkit-scrollbar-track { background:transparent; }
  ::-webkit-scrollbar-thumb { background:rgba(255,255,255,0.1); border-radius:2px; }
`;

function useUTCClock() {
  const [now, setNow] = useState(new Date());
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return `${String(now.getUTCHours()).padStart(2,'0')}:${String(now.getUTCMinutes()).padStart(2,'0')}:${String(now.getUTCSeconds()).padStart(2,'0')} UTC`;
}

function TrendChart({ target }: { target: Target }) {
  const history = target.history_30d ?? [];

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload?.length) return null;
    return (
      <div style={{ background:S.bg4, border:`1px solid ${S.borderB}`, padding:'6px 10px', fontFamily:S.mono, fontSize:10 }}>
        <div style={{ color:S.textSub, marginBottom:3 }}>{label}</div>
        {payload.map((p: any) => (
          <div key={p.name} style={{ color:'#5fe6a0' }}>{p.name}: <span style={{ color:S.text }}>{p.value}</span></div>
        ))}
      </div>
    );
  };

  if (!history.length) return null;

  // 최근 30일 데이터를 날짜 포맷해서 사용
  const data = history.map(h => ({
    date: `${h.date.slice(4,6)}/${h.date.slice(6,8)}`,
    z: parseFloat(h.z.toFixed(1)),
    goldstein: parseFloat(h.goldstein.toFixed(1)),
  }));

  return (
    <div style={{ marginBottom:12 }}>
      <div style={{ fontFamily:S.body, fontSize:9, color:S.textDim, letterSpacing:'.08em', textTransform:'uppercase', marginBottom:4 }}>
        30일 트렌드
        {target.delta_z !== undefined && (
          <span style={{ marginLeft:8, color: target.delta_z > 0 ? S.red : S.green }}>
            {target.delta_z > 0 ? '▲' : '▼'} {Math.abs(target.delta_z).toFixed(1)}
          </span>
        )}
        {target.consecutive_anomaly_days !== undefined && target.consecutive_anomaly_days > 0 && (
          <span style={{ marginLeft:8, color:S.amber }}>{target.consecutive_anomaly_days}일 연속 이상</span>
        )}
      </div>
      <ResponsiveContainer width="100%" height={90}>
        <ComposedChart data={data} margin={{ top:4, right:4, bottom:0, left:-10 }}>
          <CartesianGrid strokeDasharray="2 4" stroke="rgba(255,255,255,0.05)" vertical={false}/>
          <XAxis dataKey="date" tick={{ fontFamily:S.mono, fontSize:7, fill:S.textDim }} tickLine={false} axisLine={false} interval={4}/>
          <YAxis tick={{ fontFamily:S.mono, fontSize:8, fill:S.textDim }} tickLine={false} axisLine={false} width={40}/>
          <Tooltip content={<CustomTooltip/>}/>
          <Line dataKey="z" name="innov_z" stroke="#5fe6a0" strokeWidth={1.5} dot={false} activeDot={{ r:3 }}/>
        </ComposedChart>
      </ResponsiveContainer>
      <div style={{ display:'flex', gap:10, marginTop:3 }}>
        <div style={{ display:'flex', alignItems:'center', gap:3 }}>
          <span style={{ fontFamily:S.mono, fontSize:9, color:'#5fe6a0' }}>—</span>
          <span style={{ fontSize:9, color:S.textDim }}>Z-score</span>
        </div>
      </div>
    </div>
  );
}

function PanelHeader({ title, right }: { title: string; right?: React.ReactNode }) {
  return (
    <div className="panel-hdr">
      <span className="panel-title">{title}</span>
      {right && <span className="panel-tag">{right}</span>}
    </div>
  );
}

function ScheduleRow({ pass, rank, approved, onSelect }: {
  pass: FlatPass; rank: number; approved: boolean; onSelect: () => void;
}) {
  const urgent = pass.action_priority_label === '즉시 촬영';
  return (
    <div className={`sch-row${urgent?' urg':''}`} onClick={onSelect}
      style={{ padding:'8px 10px', borderBottom:`1px solid rgba(255,255,255,0.04)`, cursor:'pointer' }}>
      <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center' }}>
        <div style={{ display:'flex', alignItems:'center', gap:8 }}>
          <span style={{
            fontSize:9, fontWeight:600, padding:'1px 5px', borderRadius:2,
            background:'rgba(255,255,255,0.06)',
            color: S.textSub,
            border: '1px solid rgba(255,255,255,0.1)',
            fontFamily: S.mono,
          }}>{pass.sensor_type === 'optical' ? 'EO' : pass.sensor_type.toUpperCase()}</span>
          <span style={{ fontSize:12, fontWeight:500, color:'#ffffff' }}>{pass.satellite}</span>
        </div>
        <div style={{ display:'flex', alignItems:'center', gap:8 }}>
          {approved && <span style={{ fontSize:10, color:S.green }}>✓ 승인</span>}
          <div style={{ textAlign:'right' }}>
            <div style={{ fontFamily:S.mono, fontSize:9, color:S.amber, fontWeight:600 }}>{fmtMD(pass.pass_time_utc)}</div>
            <div style={{ fontFamily:S.mono, fontSize:8, color:S.textDim }}>{fmtUTC(pass.pass_time_utc)}</div>
          </div>
        </div>
      </div>
    </div>
  );
}

function ApprovedCart({ approvedPasses, onRemove }: {
  approvedPasses: FlatPass[];
  onRemove: (key: string) => void;
}) {
  if (!approvedPasses.length) return null;
  return (
    <div style={{ flexShrink:0, borderTop:`1px solid ${S.border}`, background:S.bg3 }}>
      <div style={{ display:'flex', alignItems:'center', padding:'6px 10px', borderBottom:`1px solid ${S.border}` }}>
        <span style={{ fontFamily:S.body, fontSize:10, fontWeight:600, color:S.green }}>
          ✓ 촬영 승인 목록 ({approvedPasses.length})
        </span>
      </div>
      <div style={{ maxHeight:120, overflowY:'auto' }}>
        {approvedPasses.map(p => (
          <div key={passKey(p)} className="cart-item">
            <span style={{ fontFamily:S.mono, fontSize:9, color:S.green, flexShrink:0 }}>✓</span>
            <span style={{ fontSize:11, fontWeight:500, color:'#ffffff', flex:1 }}>{p.city}</span>
            <span style={{ fontFamily:S.mono, fontSize:9, color:S.textDim }}>{p.satellite}</span>
            <span style={{ fontFamily:S.mono, fontSize:9, color:S.textDim, marginLeft:6 }}>{fmtMD(p.pass_time_utc)} {fmtUTC(p.pass_time_utc)}</span>
            <button
              onClick={() => onRemove(passKey(p))}
              style={{ background:'none', border:'none', color:S.textDim, cursor:'pointer', fontSize:12, padding:'0 2px', marginLeft:4, lineHeight:1 }}
              title="승인 취소"
            >×</button>
          </div>
        ))}
      </div>
    </div>
  );
}

function DetailPanel({ target, pass, approved, onApprove, onCancel }: {
  target: Target|null; pass: FlatPass|null; approved: boolean;
  onApprove: () => void; onCancel: () => void;
}) {
  if (!target) return (
    <div style={{ display:'flex', alignItems:'center', justifyContent:'center', height:120, color:S.textDim, fontSize:12 }}>
      항목을 선택하세요
    </div>
  );
  return (
    <div style={{ padding:12 }}>
      <div style={{ display:'flex', alignItems:'center', gap:8, marginBottom:5 }}>
        <span className={`badge b-${riskClass(target.risk_label)}`}>{riskDisplay(target.risk_label)}</span>
      </div>
      <div style={{ fontFamily:S.display, fontSize:24, fontWeight:800, marginBottom:8, lineHeight:1.2, color:'#ffffff' }}>
        {target.display_name}
      </div>
      <div style={{ fontSize:13, color:'#ffffff', lineHeight:1.7, marginBottom:10, paddingBottom:10, borderBottom:`1px solid ${S.border}` }}>
        {target.llm_message}
      </div>

      <TrendChart target={target}/>

      <div style={{ display:'grid', gridTemplateColumns:'repeat(3,1fr)', gap:4, marginBottom:10 }}>
        {([
          ['innov_z 점수', target.innov_z.toFixed(1),                                S.red],
          ['언급 수',      target.mentions_total.toLocaleString(),                   '#ffffff'],
          ['이벤트',       String(target.events_count ?? target.events ?? 0),        '#ffffff'],
        ] as [string,string,string][]).map(([l,v,c])=>(
          <div key={l} style={{ background:S.bg3, padding:'5px 7px', border:`1px solid ${S.border}` }}>
            <div style={{ fontSize:8, color:'#aaaaaa', marginBottom:2 }}>{l}</div>
            <div style={{ fontFamily:S.mono, fontSize:12, fontWeight:500, color:c }}>{v}</div>
          </div>
        ))}
      </div>

      {pass && (
        <>
          <div style={{ fontSize:9, color:'#aaaaaa', letterSpacing:'.08em', textTransform:'uppercase', marginBottom:6 }}>선택된 패스</div>
          <div style={{ background:S.bg3, border:`1px solid ${S.border}`, padding:'8px 10px', marginBottom:8 }}>
            <div style={{ display:'flex', justifyContent:'space-between', marginBottom:6 }}>
              <span style={{ fontFamily:S.mono, fontSize:11, color:'#ffffff' }}>{pass.satellite}</span>
              <span style={{ fontFamily:S.mono, fontSize:9, color:'#ffffff' }}>{fmtMD(pass.pass_time_utc)} {fmtUTC(pass.pass_time_utc)}</span>
            </div>
            <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:3 }}>
              {([
                ['Off-Nadir', pass.off_nadir_deg !== undefined ? `${pass.off_nadir_deg.toFixed(1)}°` : `${pass.max_elevation_deg.toFixed(0)}°`, '#ffffff'],
                ['운량',   `${pass.cloud_cover_pct}%`,              pass.cloud_cover_pct>70?S.amber:S.green],
                ['센서',   pass.sensor_type === 'optical' ? 'EO' : pass.sensor_type.toUpperCase(), '#ffffff'],
                ['해상도', `${pass.resolution_m}m`,                 '#ffffff'],
              ] as [string,string,string][]).map(([l,v,c])=>(
                <div key={l} style={{ background:S.bg4, padding:'4px 6px' }}>
                  <div style={{ fontSize:8, color:'#aaaaaa' }}>{l}</div>
                  <div style={{ fontFamily:S.mono, fontSize:11, fontWeight:500, color:c }}>{v}</div>
                </div>
              ))}
            </div>
          </div>
          <button
            className={`approve-btn${approved?' done':''}`}
            onClick={approved ? onCancel : onApprove}
          >
            {approved ? '✓ 촬영 승인됨 — 클릭하여 취소' : '촬영 승인 / 스케줄 확정'}
          </button>
        </>
      )}

      {/* SpaceEye-T / PlanetScope 옵션 */}
      {(target.spaceeye_option || target.planetscope_option) && (
        <div style={{ marginTop:10, marginBottom:10 }}>
          <div style={{ fontSize:9, color:'#aaaaaa', letterSpacing:'.08em', textTransform:'uppercase', marginBottom:6 }}>SIA 자산 옵션</div>
          {[target.spaceeye_option, target.planetscope_option].filter(Boolean).map((opt, i) => (
            opt && (
              <div key={i} style={{ background:S.bg3, border:`1px solid rgba(95,230,160,0.2)`, padding:'8px 10px', marginBottom:6, borderRadius:2 }}>
                <div style={{ display:'flex', justifyContent:'space-between', marginBottom:6 }}>
                  <span style={{ fontFamily:S.mono, fontSize:11, color:S.green }}>{opt.satellite}</span>
                  <span style={{ fontFamily:S.mono, fontSize:9, color:S.textSub }}>{fmtMD(opt.pass_time_utc)} {fmtUTC(opt.pass_time_utc)}</span>
                </div>
                <div style={{ display:'grid', gridTemplateColumns:'repeat(4,1fr)', gap:3 }}>
                  {([
                    ['Off-Nadir', opt.off_nadir_deg !== undefined ? `${opt.off_nadir_deg.toFixed(1)}°` : '-', '#ffffff'],
                    ['운량',      `${opt.cloud_cover_pct}%`, opt.cloud_cover_pct > 70 ? S.amber : S.green],
                    ['센서',      opt.sensor_type === 'optical' ? 'EO' : opt.sensor_type.toUpperCase(), '#ffffff'],
                    ['해상도',    `${opt.resolution_m}m`, '#ffffff'],
                  ] as [string,string,string][]).map(([l,v,c]) => (
                    <div key={l} style={{ background:S.bg4, padding:'4px 6px' }}>
                      <div style={{ fontSize:8, color:'#aaaaaa' }}>{l}</div>
                      <div style={{ fontFamily:S.mono, fontSize:11, fontWeight:500, color:c }}>{v}</div>
                    </div>
                  ))}
                </div>
              </div>
            )
          ))}
        </div>
      )}

      {target.urls_sent?.length > 0 && (
        <div style={{ marginTop:10 }}>
          <div style={{ fontSize:9, color:'#aaaaaa', letterSpacing:'.08em', textTransform:'uppercase', marginBottom:5 }}>참조 기사</div>
          {target.urls_sent.map((url,i)=>{
            const domain=(()=>{ try{ return new URL(url).hostname.replace('www.',''); }catch{ return url; }})();
            return (
              <a key={i} href={url} target="_blank" rel="noopener noreferrer" className="art-link">
                <span style={{ fontFamily:S.mono, fontSize:9, color:S.blue, flexShrink:0 }}>↗</span>
                <div style={{ minWidth:0 }}>
                  <div style={{ fontSize:9, color:'#aaaaaa', marginBottom:1 }}>{domain}</div>
                  <div style={{ fontSize:10, color:S.blue, fontWeight:500 }}>기사 원문 바로가기</div>
                </div>
              </a>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default function Dashboard() {
  const utcClock = useUTCClock();
  const [dates,    setDates]    = useState<string[]>([]);
  const [date,     setDate]     = useState('');
  const [data,     setData]     = useState<DashboardData|null>(null);
  const [selected, setSelected] = useState<string|null>(null);
  const [selPass,  setSelPass]  = useState<FlatPass|null>(null);
  const [approved, setApproved] = useState<string[]>([]);
  const [cartOpen, setCartOpen] = useState(false);
  const [mapExpanded, setMapExpanded] = useState(false);
  const [loading,  setLoading]  = useState(true);
  const [error,    setError]    = useState<string|null>(null);

  // 날짜 목록 로드
  useEffect(() => {
    fetch('/api/dates')
      .then(r => r.json())
      .then(({ dates }) => {
        if (dates?.length) {
          setDates(dates);
          setDate(dates[dates.length - 1]);
        }
      })
      .catch(() => {
        // fallback: 기존 10일치
        const fallback = [
          '20260322','20260323','20260324','20260325','20260326',
          '20260327','20260328','20260329','20260330','20260331',
        ];
        setDates(fallback);
        setDate(fallback[fallback.length - 1]);
      });
  }, []);

  useEffect(()=>{
    if (!date) return;
    setLoading(true); setError(null); setSelected(null); setSelPass(null);
    setApproved([]);
    (async()=>{
      try {
        const res = await fetch(dashboardPath(date));
        if(!res.ok) throw new Error(`HTTP ${res.status}`);
        setData(await res.json());
      } catch(e:unknown) {
        setError(e instanceof Error ? e.message : 'fetch 실패');
      } finally { setLoading(false); }
    })();
  },[date]);

  const approve = useCallback((key:string) => {
    setApproved(prev => prev.includes(key) ? prev : [...prev, key]);
  },[]);

  const cancelApprove = useCallback((key:string) => {
    setApproved(prev => prev.filter(k => k !== key));
  },[]);

  if(loading) return (
    <div style={{ background:S.bg, color:S.textDim, fontFamily:S.mono, fontSize:11, display:'flex', alignItems:'center', justifyContent:'center', height:'100vh', letterSpacing:'.1em' }}>
      LOADING...
    </div>
  );
  if(error||!data) return (
    <div style={{ background:S.bg, color:S.red, fontSize:13, padding:32 }}>⚠ {error??'데이터 없음'}</div>
  );

  const targets      = data.targets??[];
  const allPasses    = flattenPasses(targets);
  const selTarget    = targets.find(t=>t.city===selected)??null;
  const grouped      = groupPassesByTarget(targets);
  const approvedPasses = allPasses.filter(p => approved.includes(passKey(p)));

  return (
    <>
      <style>{globalStyles}</style>
      <div style={{ background:S.bg, color:S.text, fontFamily:S.body, height:'100vh', display:'flex', flexDirection:'column', padding:'10px 12px', gap:8 }}>

        {/* HEADER */}
        <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', paddingBottom:8, borderBottom:`1px solid ${S.border}`, flexShrink:0 }}>
          <div style={{ display:'flex', alignItems:'center', gap:12 }}>
            <img src="/sia-logo.jpg" alt="SIA" style={{ height:32, objectFit:'contain' }}/>
            <div style={{ width:1, height:24, background:S.border }}/>
            <div>
              <div style={{ fontFamily:S.display, fontSize:20, fontWeight:800, letterSpacing:'.06em', color:'#ffffff' }}>SI ANALYTICS</div>
              <div style={{ fontSize:10, color:S.textDim, marginTop:1 }}>위성 촬영 스케줄링 자동화 · 이란-미국 분쟁 지역</div>
            </div>
          </div>
          <div style={{ display:'flex', alignItems:'center', gap:12 }}>
            <span style={{ fontFamily:S.mono, fontSize:11, color:S.textSub, letterSpacing:'.06em' }}>{utcClock}</span>
            <div style={{ width:1, height:16, background:S.border }}/>
            <div style={{ position:'relative' }}>
              <select className="date-sel" value={date} onChange={e=>setDate(e.target.value)}>
                {dates.map(d=>(
                  <option key={d} value={d} style={{ background:S.bg3 }}>{fmtDateLabel(d)}</option>
                ))}
              </select>
              <span style={{ position:'absolute', right:6, top:'50%', transform:'translateY(-50%)', color:S.textDim, fontSize:9, pointerEvents:'none' }}>▾</span>
            </div>
          </div>
        </div>

        {/* 3분할 */}
        <Split
          className="split-horiz"
          sizes={[33, 34, 33]}
          minSize={200}
          gutterSize={5}
          direction="horizontal"
          style={{ flex:1, minHeight:0 }}
        >
          {/* 1열: 타임라인 + 스케줄 */}
          <Split
            direction="vertical"
            sizes={[35, 65]}
            minSize={80}
            gutterSize={5}
            style={{ display:'flex', flexDirection:'column', minHeight:0, height:'100%' }}
          >

            {/* 타임라인 */}
            <div className="panel" style={{ display:'flex', flexDirection:'column', minHeight:0, overflow:'hidden' }}>
              <PanelHeader title="24H 촬영 타임라인 (D+1)"/>
              <div style={{ padding:'8px 10px', overflowY:'auto', flex:1, minHeight:0 }}>
                {(() => {
                  // D+1 날짜 계산
                  const baseDate = new Date(`${date.slice(0,4)}-${date.slice(4,6)}-${date.slice(6,8)}T00:00:00Z`);
                  baseDate.setUTCDate(baseDate.getUTCDate() + 1);
                  const d1 = baseDate.toISOString().slice(0,10);

                  // 도시별로 나열 (D+1 패스)
                  const cityPasses: Record<string, FlatPass[]> = {};
                  const d1Passes: FlatPass[] = [...allPasses.filter(p => p.pass_time_utc.slice(0,10) === d1)];

                  // spaceeye_option, planetscope_option 추가
                  targets.forEach(t => {
                    [t.spaceeye_option, t.planetscope_option].filter(Boolean).forEach(opt => {
                      if (opt && opt.pass_time_utc.slice(0,10) === d1) {
                        d1Passes.push({ ...opt, city: t.city, innov_z: t.innov_z, tier: t.tier });
                      }
                    });
                  });

                  d1Passes.forEach(p => {
                    (cityPasses[p.city] ??= []).push(p);
                  });

                  return (
                    <>
                      <div style={{ fontFamily:S.mono, fontSize:8, color:S.amber, marginBottom:4 }}>{d1}</div>
                      <div style={{ display:'flex', marginLeft:110, marginBottom:4 }}>
                        {[0,4,8,12,16,20,24].map(h=>(
                          <div key={h} style={{ flex:1, fontFamily:S.mono, fontSize:8, color:S.textDim, textAlign:'center' }}>{String(h).padStart(2,'0')}</div>
                        ))}
                      </div>
                      {Object.entries(cityPasses).map(([city, passes]) => (
                        <div key={city} style={{ display:'flex', alignItems:'center', marginBottom:5, gap:5 }}>
                          <div style={{ fontSize:8, color:S.textSub, width:106, flexShrink:0, overflow:'hidden', textOverflow:'ellipsis', whiteSpace:'nowrap' }} title={city}>
                            {city}
                          </div>
                          <div style={{ flex:1, height:16, background:'rgba(255,255,255,0.04)', position:'relative', borderRadius:1 }}>
                            {passes.map((p, ei) => {
                              const hr = new Date(p.pass_time_utc).getUTCHours() + new Date(p.pass_time_utc).getUTCMinutes()/60;
                              const isApproved = approved.includes(passKey(p));
                              return (
                                <div key={ei}
                                  title={`${p.city} | ${p.satellite} | ${p.action_priority_label}`}
                                  onClick={() => { setSelected(p.city); setSelPass(p); }}
                                  style={{
                                    position:'absolute', height:'100%',
                                    left:`${(hr/24*100).toFixed(1)}%`, width:'5%', minWidth:3,
                                    borderRadius:1, display:'flex', alignItems:'center',
                                    padding:'0 2px', fontSize:7, overflow:'hidden',
                                    whiteSpace:'nowrap', cursor:'pointer',
                                    background: isApproved ? '#5fe6a0' : '#3a3a3a',
                                    color: isApproved ? '#000' : '#aaa',
                                    opacity: 1,
                                    transition: 'background 0.3s',
                                  }}>
                                  {p.city.substring(0,3)}
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      ))}
                      {Object.keys(cityPasses).length === 0 && (
                        <div style={{ fontSize:10, color:S.textDim, padding:'8px 0' }}>D+1 패스 없음</div>
                      )}
                      <div style={{ display:'flex', gap:8, marginTop:5, paddingTop:5, borderTop:`1px solid ${S.border}` }}>
                        <div style={{ display:'flex', alignItems:'center', gap:3 }}>
                          <div style={{ width:8, height:8, borderRadius:1, background:'#5fe6a0' }}/>
                          <span style={{ fontSize:8, color:S.textDim }}>촬영 승인</span>
                        </div>
                        <div style={{ display:'flex', alignItems:'center', gap:3 }}>
                          <div style={{ width:8, height:8, borderRadius:1, background:'#3a3a3a' }}/>
                          <span style={{ fontSize:8, color:S.textDim }}>미승인</span>
                        </div>
                      </div>
                    </>
                  );
                })()}
              </div>
            </div>

            {/* 스케줄 */}
            <div className="panel" style={{ flex:1, minHeight:0 }}>
              <PanelHeader title="촬영 스케줄"/>
              <div style={{ overflowY:'auto', flex:1 }}>
                {grouped.map(({ target, passes, rank }) => (
                  <div key={target.city}>
                    <div style={{ padding:'6px 10px', background:S.bg3, borderBottom:`1px solid ${S.border}`, display:'flex', alignItems:'center', gap:8, position:'sticky', top:0, zIndex:1 }}>
                      <span style={{ fontFamily:S.mono, fontSize:10, color:S.textDim, minWidth:20 }}>#{rank}</span>
                      <span className={`badge b-${riskClass(target.risk_label)}`}>{riskDisplay(target.risk_label)}</span>
                      <span style={{ fontSize:12, fontWeight:600, color:'#ffffff' }}>{target.display_name}</span>
                      <span style={{ fontFamily:S.mono, fontSize:9, color:S.textDim, marginLeft:'auto' }}>Z {target.innov_z.toFixed(1)}</span>
                    </div>
                    {passes.map((p, i) => (
                      <ScheduleRow
                        key={i} pass={p} rank={rank}
                        approved={approved.includes(passKey(p))}
                        onSelect={() => { setSelected(p.city); setSelPass(p); }}
                      />
                    ))}
                  </div>
                ))}
              </div>
            </div>
          </Split>

          {/* 2열: 지도 + 장바구니 (가운데) */}
          <div className="panel" style={{ position:'relative' }}>
            {/* 전체화면 버튼 */}
            <button
              onClick={() => setMapExpanded(true)}
              style={{
                position:'absolute', top:8, right:8, zIndex:10,
                background:'rgba(0,0,0,0.6)', border:`1px solid ${S.border}`,
                color:S.textSub, cursor:'pointer', padding:'4px 7px',
                fontSize:11, borderRadius:3, fontFamily:S.body,
              }}
              title="전체화면"
            >⛶</button>
            <div style={{ flex:1, minHeight:0, overflow:'hidden' }}>
              <LeafletMap
                targets={targets}
                selected={selected}
                onSelect={city=>{
                  setSelected(city);
                  const t = targets.find(t => t.city === city);
                  if (t?.satellite_passes?.length) {
                    const fp = { ...t.satellite_passes[0], city: t.city, innov_z: t.innov_z, tier: t.tier };
                    setSelPass(fp);
                  } else {
                    setSelPass(null);
                  }
                }}
              />
            </div>
            {/* 촬영 승인 목록 토글 */}
            <div style={{ flexShrink:0, borderTop:`1px solid ${S.border}`, background:S.bg3 }}>
              <div
                onClick={() => setCartOpen(prev => !prev)}
                style={{ display:'flex', alignItems:'center', justifyContent:'space-between', padding:'6px 10px', cursor:'pointer', borderBottom: cartOpen ? `1px solid ${S.border}` : 'none' }}
              >
                <div style={{ display:'flex', alignItems:'center', gap:6 }}>
                  <span style={{ fontFamily:S.body, fontSize:10, fontWeight:600, color:S.textSub }}>촬영 승인 목록</span>
                  {approvedPasses.length > 0 && (
                    <span style={{ fontFamily:S.mono, fontSize:9, color:S.green }}>({approvedPasses.length})</span>
                  )}
                </div>
                <span style={{ fontSize:10, color:S.textDim }}>{cartOpen ? '▲' : '▼'}</span>
              </div>
              {cartOpen && (
                approvedPasses.length === 0 ? (
                  <div style={{ padding:'8px 10px', fontSize:10, color:S.textDim }}>승인된 촬영이 없습니다</div>
                ) : (
                <div style={{ maxHeight:130, overflowY:'auto' }}>
                    {approvedPasses.map(p => (
                      <div key={passKey(p)} className="cart-item">
                        <span style={{ fontFamily:S.mono, fontSize:9, color:S.green, flexShrink:0 }}>✓</span>
                        <span style={{ fontSize:11, fontWeight:500, color:'#ffffff', flex:1 }}>{p.city}</span>
                        <span style={{ fontFamily:S.mono, fontSize:9, color:S.textDim }}>{p.satellite}</span>
                        <span style={{ fontFamily:S.mono, fontSize:9, color:S.textDim, marginLeft:6 }}>{fmtMD(p.pass_time_utc)} {fmtUTC(p.pass_time_utc)}</span>
                        <button
                          onClick={e => { e.stopPropagation(); cancelApprove(passKey(p)); }}
                          style={{ background:'none', border:'none', color:S.textDim, cursor:'pointer', fontSize:12, padding:'0 2px', marginLeft:4, lineHeight:1 }}
                          title="승인 취소"
                        >×</button>
                      </div>
                    ))}
                  </div>
                )
              )}
            </div>
          </div>

          {/* 3열: 상세 분석 */}
          <div className="panel">
            <PanelHeader title="상세 분석"/>
            <div style={{ overflowY:'auto', flex:1 }}>
              <DetailPanel
                target={selTarget}
                pass={selPass}
                approved={selPass?approved.includes(passKey(selPass)):false}
                onApprove={()=>selPass&&approve(passKey(selPass))}
                onCancel={()=>selPass&&cancelApprove(passKey(selPass))}
              />
            </div>
          </div>
        </Split>

        {/* 지도 전체화면 오버레이 */}
        {mapExpanded && (
          <div style={{
            position:'fixed', inset:0, zIndex:9999,
            background:S.bg, display:'flex', flexDirection:'column',
          }}>
            <div style={{ display:'flex', alignItems:'center', justifyContent:'space-between', padding:'8px 12px', borderBottom:`1px solid ${S.border}`, background:S.bg2, flexShrink:0 }}>
              <span style={{ fontFamily:S.body, fontSize:11, fontWeight:600, color:S.textSub }}>전술 지도</span>
              <button
                onClick={() => setMapExpanded(false)}
                style={{ background:'none', border:`1px solid ${S.border}`, color:S.text, cursor:'pointer', padding:'4px 10px', fontSize:13, borderRadius:3 }}
              >✕</button>
            </div>
            <div style={{ flex:1, minHeight:0 }}>
              <LeafletMap
                targets={targets}
                selected={selected}
                onSelect={city=>{
                  setSelected(city);
                  const t = targets.find(t => t.city === city);
                  if (t?.satellite_passes?.length) {
                    const fp = { ...t.satellite_passes[0], city: t.city, innov_z: t.innov_z, tier: t.tier };
                    setSelPass(fp);
                  } else {
                    setSelPass(null);
                  }
                  setMapExpanded(false);
                }}
              />
            </div>
          </div>
        )}

        {/* FOOTER */}
        <div style={{ flexShrink:0, borderTop:`1px solid ${S.border}`, padding:'6px 16px', display:'flex', alignItems:'center', justifyContent:'center' }}>
          <span style={{ fontFamily:S.body, fontSize:10, color:S.textDim }}>
            본 사이트는 SIA와 모두의연구소의 협력을 통한 교육과정에서 제작된 결과물입니다.
          </span>
        </div>

      </div>
    </>
  );
}
