/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        border: 'hsl(var(--tw-border) / <alpha-value>)',
        input: 'hsl(var(--tw-input) / <alpha-value>)',
        ring: 'hsl(var(--tw-ring) / <alpha-value>)',
        background: 'hsl(var(--tw-background) / <alpha-value>)',
        foreground: 'hsl(var(--tw-foreground) / <alpha-value>)',
        card: {
          DEFAULT: 'hsl(var(--tw-card) / <alpha-value>)',
          foreground: 'hsl(var(--tw-card-foreground) / <alpha-value>)',
        },
        primary: {
          DEFAULT: 'hsl(var(--tw-primary) / <alpha-value>)',
          foreground: 'hsl(var(--tw-primary-foreground) / <alpha-value>)',
        },
        secondary: {
          DEFAULT: 'hsl(var(--tw-secondary) / <alpha-value>)',
          foreground: 'hsl(var(--tw-secondary-foreground) / <alpha-value>)',
        },
        muted: {
          DEFAULT: 'hsl(var(--tw-muted) / <alpha-value>)',
          foreground: 'hsl(var(--tw-muted-foreground) / <alpha-value>)',
        },
        accent: {
          DEFAULT: 'hsl(var(--tw-accent) / <alpha-value>)',
          foreground: 'hsl(var(--tw-accent-foreground) / <alpha-value>)',
        },
        destructive: {
          DEFAULT: 'hsl(var(--tw-destructive) / <alpha-value>)',
          foreground: 'hsl(var(--tw-destructive-foreground) / <alpha-value>)',
        },
      },
      borderRadius: {
        lg: '8px',
        md: '6px',
        sm: '4px',
      },
      boxShadow: {
        soft: '0 8px 24px rgba(37, 49, 59, 0.06)',
      },
    },
  },
  plugins: [],
};
