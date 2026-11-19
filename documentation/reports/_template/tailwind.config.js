// Source for tailwind.css, the static stylesheet the builder inlines into every report.
// The pages used to load Tailwind from the Play CDN at view time; a browser that blocks that
// script showed an unstyled page. Regenerate after adding a utility class anywhere below:
//   cd documentation/reports/_template
//   npx tailwindcss@3.4.17 -c tailwind.config.js -i tailwind.input.css -o tailwind.css --minify
// shadcn/ui semantic colours are mapped onto the CSS variables in tokens.css.
module.exports = {
  darkMode: "class",
  content: ["./shell.html", "./widgets.js", "../../../scripts/reports/build_report.py"],
  theme: {
    extend: {
      fontFamily: { sans: ["Inter", "ui-sans-serif", "system-ui", "-apple-system", "Segoe UI", "Roboto", "Helvetica Neue", "Arial", "sans-serif"] },
      colors: {
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        primary: { DEFAULT: "hsl(var(--primary))", foreground: "hsl(var(--primary-foreground))" },
        secondary: { DEFAULT: "hsl(var(--secondary))", foreground: "hsl(var(--secondary-foreground))" },
        destructive: { DEFAULT: "hsl(var(--destructive))", foreground: "hsl(var(--destructive-foreground))" },
        success: "hsl(var(--success))",
        warning: "hsl(var(--warning))",
        info: "hsl(var(--info))",
        muted: { DEFAULT: "hsl(var(--muted))", foreground: "hsl(var(--muted-foreground))" },
        accent: { DEFAULT: "hsl(var(--accent))", foreground: "hsl(var(--accent-foreground))" },
        popover: { DEFAULT: "hsl(var(--popover))", foreground: "hsl(var(--popover-foreground))" },
        card: { DEFAULT: "hsl(var(--card))", foreground: "hsl(var(--card-foreground))" }
      },
      // Square corners everywhere (2026-11-19): every radius utility resolves to 0.
      borderRadius: { none: "0", sm: "0", DEFAULT: "0", md: "0", lg: "0", xl: "0", "2xl": "0", "3xl": "0", full: "0" }
    }
  }
};
