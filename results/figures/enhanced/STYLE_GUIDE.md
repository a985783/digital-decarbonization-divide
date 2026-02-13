# Nature/Science Visualization Style Guide
## Digital Decarbonization Divide Research Project

---

## 1. Color Palette

### Primary Colors
| Color Name | Hex Code | Usage |
|------------|----------|-------|
| Forest Green | `#228B22` | Primary data elements, positive effects, main trends |
| Tech Blue | `#0066CC` | Secondary data, comparison lines, reference elements |
| Gold | `#FFD700` | Accent, significance markers, highlights |

### Neutral Colors
| Color Name | Hex Code | Usage |
|------------|----------|-------|
| Near Black | `#1a1a1a` | Text, axes, borders |
| Medium Gray | `#666666` | Grid lines, secondary text |
| Light Gray | `#cccccc` | Background elements, subtle borders |
| Off-White | `#fafafa` | Figure backgrounds |

### Color Usage Guidelines
- **Confidence Intervals**: Use Forest Green at 20% opacity (`#228B2233`)
- **Heatmaps**: Use diverging palettes (RdYlGn_r) centered at zero
- **Error Bars**: Dark gray (`#1a1a1a`) with medium weight

---

## 2. Typography

### Font Family
- **Primary**: Arial, Helvetica, or DejaVu Sans
- **Fallback**: System sans-serif

### Font Sizes
| Element | Size | Weight |
|---------|------|--------|
| Figure Title | 14pt | Bold |
| Axis Labels | 12pt | Bold |
| Tick Labels | 10pt | Regular |
| Legend Text | 10pt | Regular |
| Annotations | 11pt | Bold |

### Text Guidelines
- Use sentence case for titles (not title case)
- Avoid abbreviations in axis labels when possible
- Use subscripts for chemical formulas (CO₂)
- Use log₁₀ notation for logarithmic scales

---

## 3. Figure Specifications

### Dimensions & Resolution
- **Standard Width**: 10-14 inches (for single-column: 3.5", double-column: 7.2")
- **Resolution**: 300 DPI minimum for PNG, vector PDF for publication
- **Aspect Ratio**: 4:3 or 16:9 depending on content

### Layout Principles
1. **Maximize Data-Ink Ratio**: Remove unnecessary grid lines and borders
2. **Clear Hierarchy**: Title → Axes → Data → Annotations
3. **Consistent Spacing**: Use `tight_layout()` with appropriate padding

### Spines & Borders
- Remove top and right spines
- Keep left and bottom spines at 0.8pt linewidth
- Use gray (`#666666`) for spine colors

---

## 4. Specific Figure Types

### Scatter Plots
- Point size: 15-30 depending on density
- Alpha: 0.15-0.3 for large datasets
- Edge colors: None or very subtle

### Line Plots
- Main trend: 2.5pt linewidth
- Confidence bands: Filled area with 15-20% opacity
- Reference lines: 1pt, dashed, gray

### Bar Charts
- Bar width: 0.6-0.8 of category width
- Edge color: Dark gray, 1.2pt
- Fill opacity: 0.85
- Error bars: Capsize 5pt, capthick 1.5pt

### Heatmaps
- Use diverging colormaps for signed data
- Annotate values for small matrices
- Use `square=True` for equal aspect ratio
- Colorbar shrink: 0.8 for proportion

### Density Plots
- Fill under curve with 30% opacity
- Line width: 2.5pt
- Highlight critical regions with accent color

---

## 5. Statistical Annotations

### Confidence Intervals
- Display as filled bands or error bars
- 95% confidence level (1.96 × SE)
- Label clearly in legend

### Significance Markers
- Use star (*) symbols for significant points
- Size: 100pt for visibility
- Color: Gold (`#FFD700`) with dark edge

### P-values
- Report exact values when p ≥ 0.001
- Use "p < 0.001" for smaller values
- Display in annotation box with white background

---

## 6. File Naming Convention

```
{content_descriptor}_enhanced.{format}
```

Examples:
- `divide_plot_gdp_enhanced.png`
- `gate_heatmap_energy_structure_enhanced.pdf`
- `mechanism_renewable_curve_enhanced.png`

### Formats
- **PNG**: 300 DPI for web/presentations
- **PDF**: Vector format for publication
- **Both**: Always generate both formats

---

## 7. Python Implementation

### Global Settings
```python
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.linewidth': 0.8,
    'axes.edgecolor': '#666666',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})
```

### Standard Figure Template
```python
fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

# Your plot code here

# Styling
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figure_enhanced.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('figure_enhanced.pdf', bbox_inches='tight', facecolor='white')
plt.close()
```

---

## 8. Accessibility Considerations

### Color Blindness
- Avoid red-green combinations without additional cues
- Use patterns or shapes in addition to colors
- Test with Coblis or similar tools

### Contrast
- Minimum 4.5:1 contrast ratio for text
- Minimum 3:1 for graphical elements

### Alternative Text
- Provide descriptive captions
- Include data source information

---

## 9. Generated Figures Summary

| Figure | Description | Key Features |
|--------|-------------|--------------|
| `divide_plot_gdp_enhanced` | Main relationship with CI bands | Scatter + trend + confidence band |
| `gate_plot_enhanced` | Group average treatment effects | Bar chart with error bars |
| `gate_heatmap_*_enhanced` | Multidimensional heterogeneity | Heatmaps across dimensions |
| `linear_vs_forest_enhanced` | Model comparison | Dual panel with residuals |
| `mechanism_renewable_curve_enhanced` | Interaction effects | Contour + bar comparison |
| `placebo_distribution_enhanced` | Null distribution test | Density with highlighted regions |

---

## 10. References

- Nature Portfolio: Figure Guidelines
- Science Magazine: Visual Standards
- Tufte, E. (2001). The Visual Display of Quantitative Information
- Wilke, C. (2019). Fundamentals of Data Visualization

---

*Generated: 2026-02-13*
*For: Digital Decarbonization Divide Research Project*
