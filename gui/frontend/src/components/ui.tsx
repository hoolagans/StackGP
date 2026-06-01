import React from 'react';

interface Props {
  title: string;
  children: React.ReactNode;
  className?: string;
  action?: React.ReactNode;
}

export const Card: React.FC<Props> = ({ title, children, className = '', action }) => (
  <div className={`bg-white rounded-xl border border-gray-200 shadow-sm ${className}`}>
    <div className="flex items-center justify-between px-5 py-3 border-b border-gray-100">
      <h3 className="font-semibold text-gray-800 text-sm uppercase tracking-wide">{title}</h3>
      {action && <div>{action}</div>}
    </div>
    <div className="p-5">{children}</div>
  </div>
);

interface StatProps {
  label: string;
  value: string | number | null | undefined;
  sub?: string;
  color?: 'brand' | 'green' | 'yellow' | 'red' | 'gray';
}

const statColorMap: Record<string, { wrap: string; text: string }> = {
  brand: { wrap: 'bg-brand-50 border-brand-100', text: 'text-brand-700' },
  green: { wrap: 'bg-green-50 border-green-100', text: 'text-green-700' },
  yellow: { wrap: 'bg-yellow-50 border-yellow-100', text: 'text-yellow-700' },
  red:    { wrap: 'bg-red-50 border-red-100',     text: 'text-red-700'    },
  gray:   { wrap: 'bg-gray-50 border-gray-100',   text: 'text-gray-700'   },
};

export const StatBadge: React.FC<StatProps> = ({ label, value, sub, color = 'brand' }) => {
  const c = statColorMap[color] ?? statColorMap.brand;
  const formatted = value == null
    ? '—'
    : typeof value === 'number'
      ? (Number.isInteger(value) ? value.toLocaleString() : value.toFixed(4))
      : value;
  return (
    <div className={`rounded-lg border px-4 py-3 ${c.wrap}`}>
      <div className="text-xs text-gray-500 font-medium mb-0.5">{label}</div>
      <div className={`text-xl font-bold ${c.text}`}>
        {formatted}
      </div>
      {sub && <div className="text-xs text-gray-400 mt-0.5">{sub}</div>}
    </div>
  );
};

interface BadgeProps { text: string; variant?: 'green' | 'blue' | 'yellow' | 'red' | 'gray' }
export const Badge: React.FC<BadgeProps> = ({ text, variant = 'gray' }) => {
  const cls: Record<string, string> = {
    green: 'bg-green-100 text-green-700',
    blue: 'bg-blue-100 text-blue-700',
    yellow: 'bg-yellow-100 text-yellow-700',
    red: 'bg-red-100 text-red-700',
    gray: 'bg-gray-100 text-gray-600',
  };
  return <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${cls[variant]}`}>{text}</span>;
};

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'danger' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  icon?: React.ReactNode;
}
export const Button: React.FC<ButtonProps> = ({
  variant = 'primary', size = 'md', loading, icon, children, className = '', ...rest
}) => {
  const base = 'inline-flex items-center gap-2 font-medium rounded-lg transition-all focus:outline-none focus:ring-2 focus:ring-offset-1 disabled:opacity-50 disabled:cursor-not-allowed';
  const sizes = { sm: 'px-3 py-1.5 text-xs', md: 'px-4 py-2 text-sm', lg: 'px-6 py-3 text-base' };
  const variants = {
    primary: 'bg-brand-600 hover:bg-brand-700 text-white focus:ring-brand-500',
    secondary: 'bg-white hover:bg-gray-50 text-gray-700 border border-gray-300 focus:ring-gray-300',
    danger: 'bg-red-600 hover:bg-red-700 text-white focus:ring-red-500',
    ghost: 'hover:bg-gray-100 text-gray-600 focus:ring-gray-300',
  };
  return (
    <button className={`${base} ${sizes[size]} ${variants[variant]} ${className}`} disabled={loading} {...rest}>
      {loading ? <span className="w-4 h-4 border-2 border-current border-t-transparent rounded-full animate-spin" /> : icon}
      {children}
    </button>
  );
};

export const Select: React.FC<React.SelectHTMLAttributes<HTMLSelectElement> & { label?: string }> = ({
  label, className = '', ...rest
}) => (
  <label className="block">
    {label && <span className="text-xs font-medium text-gray-600 mb-1 block">{label}</span>}
    <select
      className={`w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm text-gray-800 focus:outline-none focus:ring-2 focus:ring-brand-400 ${className}`}
      {...rest}
    />
  </label>
);

export const Input: React.FC<React.InputHTMLAttributes<HTMLInputElement> & { label?: string }> = ({
  label, className = '', ...rest
}) => (
  <label className="block">
    {label && <span className="text-xs font-medium text-gray-600 mb-1 block">{label}</span>}
    <input
      className={`w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm text-gray-800 focus:outline-none focus:ring-2 focus:ring-brand-400 ${className}`}
      {...rest}
    />
  </label>
);

export const Spinner: React.FC<{ className?: string }> = ({ className = 'w-6 h-6' }) => (
  <div className={`border-2 border-brand-200 border-t-brand-600 rounded-full animate-spin ${className}`} />
);

export const EmptyState: React.FC<{ icon?: string; title: string; subtitle?: string }> = ({
  icon = '📂', title, subtitle
}) => (
  <div className="flex flex-col items-center justify-center py-16 text-center">
    <div className="text-4xl mb-3">{icon}</div>
    <div className="text-gray-600 font-medium">{title}</div>
    {subtitle && <div className="text-sm text-gray-400 mt-1">{subtitle}</div>}
  </div>
);
