# Visual QA verdict

Measured in Chromium (Playwright build 1248) against report.html, deterministic in-page probe; screenshots kept as shot-light-1280.png and shot-dark-390.png.

## light, 1280px
- horizontal overflow: 0px - PASS
- body text contrast on page background: 16.12:1 - PASS (WCAG 4.5 floor)
- h1 contrast: 16.12:1 - PASS
- secondary/muted text contrast: 4.86:1 - PASS
- chart label collisions (bounding-box overlap of svg text): 0 - PASS
- quote lines render as filled q elements: PASS
- table fits inside main column: PASS
- main column width: 800px (760px max-width + 20px padding) - context

## dark, 390px
- horizontal overflow: 0px - PASS
- body text contrast on page background: 16.44:1 - PASS
- h1 contrast: 16.44:1 - PASS
- secondary/muted text contrast: 6.23:1 - PASS
- chart label collisions: 0 - PASS
- quote lines render as filled q elements: PASS
- table fits inside main column: PASS
- main column width: 375px (viewport minus padding) - context

Overall: PASS - 0 defects.
