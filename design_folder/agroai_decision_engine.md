---
name: AgroAI Decision Engine
colors:
  surface: '#f8f9ff'
  surface-dim: '#d0dbed'
  surface-bright: '#f8f9ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#eff4ff'
  surface-container: '#e6eeff'
  surface-container-high: '#dee9fc'
  surface-container-highest: '#d9e3f6'
  on-surface: '#121c2a'
  on-surface-variant: '#424844'
  inverse-surface: '#27313f'
  inverse-on-surface: '#eaf1ff'
  outline: '#727974'
  outline-variant: '#c1c8c2'
  surface-tint: '#456555'
  primary: '#06271a'
  on-primary: '#ffffff'
  primary-container: '#1e3d2f'
  on-primary-container: '#86a895'
  inverse-primary: '#accebb'
  secondary: '#296b3c'
  on-secondary: '#ffffff'
  secondary-container: '#adf3b8'
  on-secondary-container: '#307142'
  tertiary: '#11270c'
  on-tertiary: '#ffffff'
  tertiary-container: '#273d20'
  on-tertiary-container: '#8ea882'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#c7ebd6'
  primary-fixed-dim: '#accebb'
  on-primary-fixed: '#012114'
  on-primary-fixed-variant: '#2e4d3e'
  secondary-fixed: '#adf3b8'
  secondary-fixed-dim: '#92d69d'
  on-secondary-fixed: '#00210b'
  on-secondary-fixed-variant: '#0a5226'
  tertiary-fixed: '#cfebc1'
  tertiary-fixed-dim: '#b3cea6'
  on-tertiary-fixed: '#0b2006'
  on-tertiary-fixed-variant: '#364d2e'
  background: '#f8f9ff'
  on-background: '#121c2a'
  surface-variant: '#d9e3f6'
typography:
  display-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.02em
  display-lg-mobile:
    fontFamily: Inter
    fontSize: 26px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.01em
  headline-lg:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.015em
  headline-md:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: -0.01em
  headline-sm:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '600'
    lineHeight: 24px
    letterSpacing: -0.005em
  body-lg:
    fontFamily: Inter
    fontSize: 15px
    fontWeight: '400'
    lineHeight: 24px
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  body-sm:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 18px
  label-md:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.01em
  label-sm:
    fontFamily: Inter
    fontSize: 11px
    fontWeight: '600'
    lineHeight: 14px
    letterSpacing: 0.03em
  data-mono:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '500'
    lineHeight: 18px
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  gutter: 1.25rem
  gutter-sm: 0.75rem
  gutter-lg: 1.5rem
  margin: 1.5rem
  margin-sm: 1rem
  margin-lg: 2rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 0.75rem
  space-lg: 1.25rem
  space-xl: 1.75rem
  space-2xl: 2.5rem
---

## Brand & Style

The design system embodies a disciplined, modern, and data-first philosophy tailored for precision agriculture and agronomic decision intelligence. Designed for agronomists, farm operators, and enterprise agricultural analysts, the interface evokes calm authority, pragmatic competence, and trust under operational pressure.

The aesthetic fuses high-end Nordic functionalism with enterprise field utility. It avoids tech clichés: no neon greens, no decorative cybernetic nodes, and no gratuitous gradients. Instead, visual prestige comes from structural precision, impeccable typographical hierarchy, organic yet muted natural tones, and crisp data density. Surfaces feel substantial, tactile, and grounded—prioritizing daylight legibility and seamless translation from command center desktop rigs to rugged field tablets.

## Colors

The palette is strictly calibrated for high ambient-light performance, daylight readability, and prolonged focus without cognitive fatigue. The UI operates exclusively in light mode.

- **Primary (`#1E3D2F`)**: Deep Forest Green. Serves as the primary operational anchor: primary navigation frames, critical action buttons, active high-order states, and primary KPI focus points.
- **Secondary (`#2E6F40`)**: Natural Leaf Green. Used for active analytical accents, progress indicators, confirmed system recommendations, and primary telemetry highlights.
- **Tertiary (`#657E5B`)**: Muted Olive / Sage. Provides supporting data context, secondary charts, inactive timeline nodes, and peripheral metadata tags.
- **Canvas (`#F8F9F5`)**: Warm Off-White / Soft Cream. A grounded, non-glare architectural backdrop that separates the system from sterile clinical software.
- **Surface (`#FFFFFF`)**: Pure White. Dedicated strictly to content cards, data tables, filter bars, and modal overlays, creating effortless contrast against the canvas.
- **Subtle Accents (`#EAE6DF`)**: Soft Beige / Straw. Applied to neutral tags, subtle table column highlights, and contextual dividers.
- **Hairlines & Borders (`#E5E7EB`)**: Soft Cool Gray. Thin 1px structural dividing lines ensuring clear container boundaries without clutter.
- **Typography Neutrals (`#1F2937` & `#374151`)**: Crisp Slate / Charcoal. High-contrast, optical-grade text tokens ensuring zero eye strain across varying display brightness.

## Typography

Typography is systematic, utilitarian, and architected for high information density. Inter is implemented across all roles, paired with native tabular figures (`tnum`, `zero`) to ensure telemetry charts, yield projections, and sensor logs align cleanly across rows and columns.

Headings remain understated; oversized display treatments are avoided to maximize immediate data visibility above the fold. Labels use slightly tracked uppercase or medium-weight title casing for categorical tags and field metrics. Numerical data tables and spatial readings inherit the dedicated `data-mono` styling variant (leveraging Inter's OpenType tabular alignment) to maintain strict vertical scanning parity.

## Layout & Spacing

The layout model is anchored on an 8pt base grid with a 4pt sub-unit for tight micro-alignments in telemetry badges and data cells. 

- **Desktop & Large Displays (≥1280px)**: 12-column layout with a fixed or collapsible 240px navigation dock. Page content expands within a fluid grid container using a max-width of 1680px, flanked by `margin-lg` (32px) and interior column `gutter-lg` (24px).
- **Tablet & Split Field Consoles (768px - 1279px)**: 8-column layout utilizing `margin` (24px) and `gutter` (20px). Side panels tuck into slide-over drawers or modular split views to maintain primary map/chart continuity.
- **Mobile Handheld (<768px)**: 4-column layout utilizing `margin-sm` (16px) and `gutter-sm` (12px). Multi-column comparison tables collapse into accordion lists or horizontal carousels with fixed primary keys.

Card interiors use strict internal padding steps: `space-md` (12px) for condensed telemetry units and `space-lg` (20px) for analytical dashboards and decision cards.

## Elevation & Depth

Visual hierarchy relies on structural, tonal layering and hairline delineation rather than drop shadows. 

- **Surface Separation**: Pure White (`#FFFFFF`) containers rest on the Warm Off-White (`#F8F9F5`) canvas, separated by crisp 1px borders colored in Soft Cool Gray (`#E5E7EB`). 
- **Subtle Surface Depth**: Ambient elevation is minimal and natural. Standard cards apply no blur shadow, only the 1px border. Interactive cards and hovering grid items take on an ultra-diffused, ground-tinted shadow: `0 2px 8px -2px rgba(30, 61, 47, 0.05), 0 1px 3px -1px rgba(0, 0, 0, 0.04)`.
- **Floating Overlays & Modals**: Contextual filters, drawer panels, and flyouts use a measured elevation: `0 12px 28px -4px rgba(30, 61, 47, 0.08), 0 4px 10px -2px rgba(0, 0, 0, 0.03)` wrapped in a 1px border (`#E5E7EB`). 
- **Active / Focused Depth**: Active states avoid halo glows or fluorescent rings; they employ an internal 2px inset outline using Deep Forest Green (`#1E3D2F`) or Natural Leaf Green (`#2E6F40`).

## Shapes

The interface adopts a disciplined, moderate corner radius that feels architectural and robust:
- **Base Components**: Inputs, buttons, segmented controls, table headers, and badges maintain a unified 4px (`rounded-sm`) to 6px (`rounded-md`) radius.
- **Cards & Data Surfaces**: Dashboard widgets, analytical panels, and modal containers use 8px (`rounded-lg`).
- **Pill Exceptions**: Status indicators, category chips, and live sensor flags use an 8px radius or semi-compact tag format. Circular or bulbous capsule buttons are avoided entirely to maintain enterprise-grade rigor.

## Components

### Buttons
- **Primary**: Solid Deep Forest Green (`#1E3D2F`) background, Pure White (`#FFFFFF`) text, 6px radius, `0 1px 2px rgba(0,0,0,0.05)` shadow. Hover transitions to `#162E23`. Height: 36px (default), 32px (compact data grid).
- **Secondary**: Pure White (`#FFFFFF`) background, 1px border in `#E5E7EB`, text in Crisp Slate (`#1F2937`). Hover introduces subtle background shift to `#F8F9F5` and border `#D1D5DB`.
- **Subtle / Ghost**: Transparent background, text in `#2E6F40` or `#374151`. Hover background `#F0F3EC`.
- **Destructive**: Subdued brick red text and border; transitions to solid `#991B1B` with white text only upon secondary confirmation.

### Chips & Badges
- **Status Badges**: Semi-compact (height: 22px), 4px radius, 11px semi-bold text with tabular numbers. Built using tinted, non-fluorescent fills:
  - *Optimal / Healthy*: Background `#EAF3ED`, Text `#2E6F40`, 1px border `#C8E2D1`.
  - *Advisory / Alert*: Background `#FDF6E2`, Text `#8D6B14`, 1px border `#F5E5B8`.
  - *Critical Action*: Background `#FDF2F2`, Text `#991B1B`, 1px border `#F8D7DA`.
  - *Neutral / Inactive*: Background `#EAE6DF`, Text `#374151`, 1px border `#DDD7CD`.

### Input Fields & Selectors
- Background: Pure White (`#FFFFFF`), Height: 36px, Border: 1px `#E5E7EB`, Radius: 6px.
- Text: 14px `#1F2937`, Placeholder: `#9CA3AF`.
- Focus State: 1px border `#1E3D2F` with an additional 1px solid ring in `#1E3D2F` (no fuzzy ambient glow).
- Affixes: Unit markers (e.g., `kg/ha`, `pH`, `°C`, `mm`) set in Muted Sage (`#657E5B`) with `Inter` tabular font.

### Cards & Data Panels
- Background: Pure White (`#FFFFFF`).
- Border: 1px solid `#E5E7EB`.
- Header: Separated by a 1px border-bottom (`#F3F4F6`), padding: 12px 16px. Title in 14px semi-bold `#1F2937`.
- Metric Layout: Prominent numeric value (24px/32px tabular weight 600) anchored directly above contextual comparison micro-text (e.g., `+4.2% vs 5-yr yield avg`).

### Tables & Data Grids
- Header: Light Warm Off-White (`#F8F9F5`), 11px uppercase tracked labels (`#657E5B`), 1px solid bottom border (`#E5E7EB`).
- Rows: 40px default height for compact data density, alternating hover state (`#FAFAF8`).
- Cell Values: Tabular figures aligned right for numerical telemetry, left for field metadata, centered for categorical status flags.

### Checkboxes & Radio Controls
- Base: 16px square (checkbox) or circle (radio), 1px solid `#D1D5DB`, background `#FFFFFF`.
- Checked State: Solid `#1E3D2F` with an optical-white vector mark; radius for checkboxes is 4px.

### Domain-Specific Components
- **Field Recommendation Banners**: Muted off-white card edged with a 4px left-border anchor in `#2E6F40`. Displays algorithmic confidence score, actionable agronomic intervention, and inline batch-action trigger.
- **NDVI & Telemetry Mini-Scales**: Compact 6px horizontal segmented bars using categorical step swatches (Olive to Forest Green) with precise tick indicators, eschewing smooth neon gradients.