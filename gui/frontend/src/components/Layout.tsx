import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  Upload, Settings, BarChart2, Brain, LineChart, Zap, RefreshCw,
} from 'lucide-react';
import { resetSession } from '../api/client';

const NAV = [
  { to: '/import',    label: 'Import',     icon: Upload     },
  { to: '/process',   label: 'Process',    icon: Settings   },
  { to: '/explore',   label: 'Explore',    icon: BarChart2  },
  { to: '/model',     label: 'Model',      icon: Brain      },
  { to: '/analysis',  label: 'Analysis',   icon: LineChart  },
  { to: '/predict',   label: 'Predict',    icon: Zap        },
];

const Layout: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const handleReset = async () => {
    if (confirm('Reset all session data?')) {
      await resetSession();
      window.location.href = '/import';
    }
  };

  return (
    <div className="flex h-screen bg-gray-50 overflow-hidden">
      {/* Sidebar */}
      <aside className="w-56 bg-gray-900 flex flex-col shrink-0">
        {/* Logo */}
        <div className="px-5 py-5 border-b border-gray-700">
          <div className="text-white font-bold text-lg leading-tight">
            Stack<span className="text-brand-400">GP</span>
          </div>
          <div className="text-gray-400 text-xs mt-0.5">Data Modeling Studio</div>
        </div>

        {/* Nav */}
        <nav className="flex-1 px-3 py-4 space-y-1">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  isActive
                    ? 'bg-brand-600 text-white'
                    : 'text-gray-400 hover:bg-gray-800 hover:text-white'
                }`
              }
            >
              <Icon size={16} />
              {label}
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="px-3 py-4 border-t border-gray-700">
          <button
            onClick={handleReset}
            className="flex items-center gap-2 text-gray-400 hover:text-white text-xs px-3 py-2 rounded-lg hover:bg-gray-800 w-full transition-colors"
          >
            <RefreshCw size={13} />
            Reset Session
          </button>
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 overflow-auto">
        {children}
      </main>
    </div>
  );
};

export default Layout;
