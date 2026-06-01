import React from 'react';

interface Props {
  children: React.ReactNode;
}

interface State {
  hasError: boolean;
}

class ErrorBoundary extends React.Component<Props, State> {
  state: State = { hasError: false };

  static getDerivedStateFromError(): State {
    return { hasError: true };
  }

  componentDidCatch(error: unknown) {
    console.error('Unhandled render error', error);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center p-6 bg-gray-50">
          <div className="max-w-md w-full bg-white border border-gray-200 rounded-xl p-6 text-center">
            <div className="text-2xl mb-3">⚠️</div>
            <h1 className="text-lg font-semibold text-gray-800">Something went wrong</h1>
            <p className="text-sm text-gray-500 mt-2">
              The interface crashed while rendering. Please refresh the page and try again.
            </p>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

export default ErrorBoundary;
