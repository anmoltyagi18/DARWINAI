'use client';

import React from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { 
  BarChart3, 
  LayoutDashboard, 
  History, 
  Zap, 
  Settings, 
  LogOut, 
  ShieldCheck, 
  Layers,
  Search,
  PieChart
} from 'lucide-react';
import { motion } from 'framer-motion';

const NAV_ITEMS = [
  { icon: <LayoutDashboard size={20} />, label: 'Dashboard', href: '/dashboard' },
  { icon: <Zap size={20} />, label: 'Live Trading', href: '/live' },
  { icon: <Layers size={20} />, label: 'Strategy Lab', href: '/lab' },
  { icon: <BarChart3 size={20} />, label: 'Backtest', href: '/backtest' },
  { icon: <PieChart size={20} />, label: 'Portfolio', href: '/portfolio' },
  { icon: <History size={20} />, label: 'Transactions', href: '/history' },
];

export const Sidebar: React.FC = () => {
  const pathname = usePathname();

  return (
    <aside className="w-64 bg-[#0b0f14] border-r border-white/5 flex flex-col h-screen overflow-hidden">
      {/* Brand Header */}
      <div className="p-6 border-b border-white/5 bg-black/40">
        <div className="flex items-center space-x-3">
           <div className="w-10 h-10 bg-cyan-500 rounded-xl flex items-center justify-center shadow-[0_0_20px_rgba(6,182,212,0.4)]">
              <ShieldCheck className="text-black" size={24} />
           </div>
           <div>
              <h1 className="text-xl font-black italic tracking-tighter text-white">AIGOFIN.X</h1>
              <span className="text-[10px] text-cyan-400 font-mono tracking-widest uppercase opacity-60">Quant Kernel v4</span>
           </div>
        </div>
      </div>

      {/* Navigation Groups */}
      <div className="flex-1 py-8 px-4 space-y-8 overflow-y-auto custom-scrollbar">
        <div>
           <h4 className="px-4 text-[10px] font-bold text-white/20 uppercase tracking-[0.2em] mb-4">Core Ecosystem</h4>
           <nav className="space-y-1">
              {NAV_ITEMS.map((item) => {
                const isActive = pathname === item.href;
                return (
                  <Link 
                    key={item.href} 
                    href={item.href}
                    className={`group relative flex items-center px-4 py-3 rounded-lg transition-all duration-300 ${
                      isActive 
                        ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 shadow-lg shadow-cyan-500/5' 
                        : 'text-gray-400 hover:text-white hover:bg-white/5'
                    }`}
                  >
                    {isActive && (
                      <motion.div 
                        layoutId="nav-glow"
                        className="absolute left-0 w-1 h-6 bg-cyan-400 rounded-r shadow-[0_0_10px_rgba(6,182,212,1)]"
                        transition={{ type: "spring", stiffness: 300, damping: 30 }}
                      />
                    )}
                    <span className={`mr-3 transition-transform group-hover:scale-110 ${isActive ? 'text-cyan-400' : 'text-gray-500'}`}>
                      {item.icon}
                    </span>
                    <span className="text-sm font-medium tracking-tight uppercase">{item.label}</span>
                  </Link>
                );
              })}
           </nav>
        </div>

        <div>
           <h4 className="px-4 text-[10px] font-bold text-white/20 uppercase tracking-[0.2em] mb-4">System</h4>
           <nav className="space-y-1">
              <SidebarLink icon={<Settings size={18} />} label="Settings" />
              <SidebarLink icon={<LogOut size={18} />} label="Logout" danger />
           </nav>
        </div>
      </div>

      {/* Profile / Context Footer */}
      <div className="p-4 bg-black/40 border-t border-white/5">
         <div className="flex items-center p-3 rounded-xl bg-white/5 border border-white/5">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-cyan-500 to-purple-600 border border-white/20"></div>
            <div className="ml-3">
               <div className="text-xs font-bold text-white uppercase tracking-tighter">Institutional Account</div>
               <div className="text-[10px] text-green-400 animate-pulse font-mono uppercase tracking-[0.1em]">Authenticated</div>
            </div>
         </div>
      </div>
    </aside>
  );
};

const SidebarLink = ({ icon, label, danger }: any) => (
  <button className={`w-full flex items-center px-4 py-2.5 rounded-lg text-sm font-medium transition-all ${
    danger ? 'text-red-400 hover:bg-red-400/10' : 'text-gray-500 hover:text-white hover:bg-white/5'
  }`}>
    <span className="mr-3">{icon}</span>
    <span className="uppercase tracking-tight">{label}</span>
  </button>
);
