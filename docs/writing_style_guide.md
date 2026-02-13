# Writing Style Guide for Digital Decarbonization Paper

## Document Purpose
This style guide ensures consistency across all revisions, co-author contributions, and future extensions of the paper. Adherence to these guidelines maintains the professional standards expected in top-tier economics journals.

---

## 1. Terminology Standards

### Core Concepts

| Term | First Mention | Subsequent Uses | Formatting |
|------|---------------|-----------------|------------|
| Domestic Digital Capacity | Domestic Digital Capacity (DCI) | DCI | Italicized: `\textit{DCI}` |
| External Digital Specialization | External Digital Specialization (EDS) | EDS | Italicized: `\textit{EDS}` |
| Sweet Spot | "Sweet Spot" | sweet spot or Sweet Spot | Bold in titles/abstract; normal in text |
| Two-Dimensional Digitalization | Two-Dimensional Digitalization | the framework | Capitalized as proper noun |
| Model Ladder | Model Ladder | the ladder | Capitalized as proper noun |
| Digital Decarbonization Divide | Digital Decarbonization Divide | the divide | Capitalized as proper noun |
| Digital-EKC | Digital-EKC | - | Hyphenated, capitalized |

### Variable Names

| Variable | Abbreviation | Unit | Formatting |
|----------|--------------|------|------------|
| CO2 Emissions | - | metric tons per capita | CO$_2$ (use math mode subscript) |
| GDP per capita | GDP pc | constant 2015 US$ | GDP per capita |
| Renewable Energy Share | - | percentage | renewable energy share (%) |
| Control of Corruption | - | WGI index | Control of Corruption |

### Methodological Terms

| Term | Usage Notes |
|------|-------------|
| Causal Forest | Capitalized; refers to Athey et al. (2019) method |
| Double Machine Learning | Capitalized; can abbreviate to DML after first use |
| Group Average Treatment Effect | GATE (abbreviate after first use) |
| Conditional Average Treatment Effect | CATE (abbreviate after first use) |
| Average Treatment Effect | ATE (abbreviate after first use) |
| Leave-One-Country-Out | LOCO (abbreviate after first use) |

---

## 2. Grammar and Style Rules

### Voice

**Prefer Active Voice**
- ✅ "We demonstrate that..."
- ✅ "The analysis reveals..."
- ❌ "It was demonstrated that..."
- ❌ "The results were found to..."

**Exceptions** (Passive voice acceptable):
- When the actor is unknown: "The data were collected from WDI"
- When emphasizing the action over the actor: "The sample was restricted to..."
- In methodological descriptions: "The model is estimated using..."

### Tense

| Section | Tense | Example |
|---------|-------|---------|
| Abstract | Present | "This paper demonstrates..." |
| Introduction | Present/Perfect | "Previous studies have found... We show..." |
| Literature Review | Present | "The EKC literature posits..." |
| Methodology | Present | "We implement CausalForestDML..." |
| Results | Past | "The analysis revealed..." |
| Discussion | Present | "These findings suggest..." |
| Conclusion | Present | "This paper introduces..." |

### Person

- Use **first person plural** ("we") throughout
- Avoid "I" (single author) or "the author"
- Avoid "this paper" except in abstract and conclusion

### Numbers and Units

**Numbers**
- Spell out one to nine; use numerals for 10+
- Always use numerals with units: "5 metric tons"
- Use commas for thousands: "1,000 bootstrap iterations"
- Decimal places: Match precision to measurement (usually 2-3 decimal places for coefficients)

**Units**
- Always specify units on first mention
- Use standard abbreviations: tons/capita, %, US$
- Use "metric tons" not "tonnes" (American English convention)

### Punctuation

**Serial Comma (Oxford Comma)**
- Always use: "X, Y, and Z" not "X, Y and Z"

**Hyphenation**
- Compound adjectives: "high-income countries", "middle-income economies"
- But: "countries with high income" (no hyphen when not modifying noun)

**Em-dashes**
- Use for parenthetical statements: "---raising a critical question---"
- No spaces around em-dashes

**Quotation Marks**
- Use double quotes for terminology: "Sweet Spot"
- Use single quotes for scare quotes or nested quotations

---

## 3. Citation Format

### In-Text Citations

**Standard citation**
- "Previous studies validate the income-emission relationship \citep{Stern2004, Dinda2004}."

**Citation as noun**
- "\citet{Athey2019} demonstrate that..."

**Multiple citations**
- Order chronologically: "\citep{Grossman1995, Stern2004, Athey2019}"
- Use semicolons for multiple with notes: "\citep[see also][]{Stern2004, Dinda2004}"

### Citation Style

- Use author-year format (APA-like)
- Use `natbib` package with `apalike` bibliography style
- Ensure all citations have corresponding entries in `references.bib`

---

## 4. Table and Figure Formatting

### Tables

**Standard Structure**
```latex
\begin{table}[H]
\centering
\caption{Descriptive Caption}
\label{tab:shortname}
\begin{tabular}{lcc}
\toprule
Column 1 & Column 2 & Column 3 \\
\midrule
Row 1 & Value & Value \\
\bottomrule
\end{tabular}
\end{table}
```

**Rules**
- Always use `booktabs` rules (`\toprule`, `\midrule`, `\bottomrule`)
- Never use vertical rules
- Align numbers by decimal point where possible
- Use `\multicolumn` for spanning headers
- Place notes below table using `\textit{Note: ...}`

### Figures

**Standard Structure**
```latex
\begin{figure}[H]
\centering
\includegraphics[width=0.85\linewidth]{path/to/figure.png}
\caption{Descriptive caption explaining what the figure shows}
\label{fig:shortname}
\end{figure}
```

**Caption Guidelines**
- Start with descriptive title
- Explain key patterns visible in figure
- Reference specific colors/elements if necessary
- Keep under 3 sentences

---

## 5. Mathematical Notation

### Equations

**Numbered equations** for important results:
```latex
\begin{equation}
\tau(x) = \mathbb{E}[Y(1) - Y(0) | X = x]
\end{equation}
```

**Unnumbered equations** for derivations:
```latex
\[
Y = \beta_1 T + \epsilon
\]
```

### Notation Standards

| Symbol | Meaning | LaTeX |
|--------|---------|-------|
| $\tau$ | Treatment effect | `\tau` |
| $\tau(x)$ | Conditional Average Treatment Effect | `\tau(x)` |
| $Y$ | Outcome (CO2 emissions) | `Y` |
| $T$ | Treatment (DCI) | `T` |
| $X$ | Moderators | `X` |
| $W$ | Controls | `W` |
| $\beta$ | Coefficient | `\beta` |
| $\epsilon$ | Error term | `\epsilon` |
| $\mathbb{E}$ | Expectation | `\mathbb{E}` |

### Statistical Reporting

**Standard format**
- Coefficients: Report point estimate, standard error, confidence interval
- Example: "$-1.73$ (SE: $0.588$, 95\% CI: $[-2.882, -0.578]$)"

**Significance**
- Use asterisks for tables: *p < 0.1, **p < 0.05, ***p < 0.01
- In text: "statistically significant at the 1\% level"
- Report exact p-values when possible: ($p = 0.003$)

---

## 6. Section Headings

### Hierarchy

```
\section{Main Section}           % Level 1: All caps in output
\subsection{Subsection}          % Level 2: Title case
\subsubsection{Sub-subsection}   % Level 3: Title case, italicized
\paragraph{Paragraph}            % Level 4: Run-in heading
```

### Capitalization

- **Level 1**: Title Case ("Empirical Results")
- **Level 2+**: Title Case
- **Exceptions**: Prepositions < 4 letters lowercase: "of", "in", "on", "to"

### Numbering

- Number all sections except Acknowledgments, References, Appendices
- Use automatic LaTeX numbering (`\section{}`, not `\section*{}`)

---

## 7. Special Formatting

### Abbreviations

**First use**: Spell out fully with abbreviation in parentheses
- "Group Average Treatment Effects (GATEs)"
- "Domestic Digital Capacity (DCI)"

**Subsequent use**: Abbreviation only
- "GATEs reveal..."
- "DCI reduces..."

### Emphasis

**Bold** for:
- Key findings
- Policy implications
- Section takeaways
- "Sweet Spot" when referring to the concept

**Italics** for:
- DCI and EDS (as defined terms)
- Latin phrases: "ex ante", "ceteris paribus"
- Emphasis in quotations

**Underline**: Avoid in academic writing

### Quotations

**Block quotes** (for quotations > 40 words):
```latex
\begin{quote}
Text of quotation...
\end{quote}
```

**Inline quotes**: Use quotation marks, not block format

---

## 8. Common Errors to Avoid

### Language

| Error | Correction |
|-------|------------|
| "Data is" | "Data are" (plural) |
| "Proven" (adj) | "Proved" (verb past tense) |
| "Comprised of" | "Comprises" or "is composed of" |
| "Due to" (at beginning) | "Because of" |
| "Whilst" | "While" (American English) |
| " amongst" | " among" |

### Economics-Specific

| Error | Correction |
|-------|------------|
| "Elasticity is 0.5%" | "Elasticity is 0.5" (unitless) |
| "GDP growth was negative" | "GDP growth was negative" (okay) or "GDP contracted" |
| "Statistically significant" without level | Always specify: "at the 5% level" |

### LaTeX

| Error | Correction |
|-------|------------|
| CO2 | CO$_2$ (use math mode subscript) |
| "..." | ``...'' (use proper LaTeX quotes) |
| 95% | 95\% (escape percent sign) |
| p < 0.05 | $p < 0.05$ (use math mode) |

---

## 9. Journal-Specific Adaptations

### For General Economics Journals (AER, QJE, JPE)

- Follow above guidelines exactly
- Emphasize causal identification
- Include detailed robustness checks

### For Environmental Economics Journals (JEEM, REEP)

- Expand policy implications section
- Add more environmental context in introduction
- Consider adding non-technical summary

### For Policy Journals

- Reduce methodological detail (move to appendix)
- Expand policy toolkit section
- Add executive summary

---

## 10. Checklist Before Submission

### Content
- [ ] All hypotheses tested and reported
- [ ] Robustness checks complete
- [ ] Limitations acknowledged
- [ ] Policy implications clear

### Formatting
- [ ] All tables use `booktabs` (no vertical lines)
- [ ] All figures have clear captions
- [ ] All equations numbered if referenced
- [ ] All citations in bibliography

### Language
- [ ] No passive voice > 20% of sentences
- [ ] Terminology consistent throughout
- [ ] No undefined abbreviations
- [ ] JEL codes appropriate

### Technical
- [ ] Paper compiles without errors
- [ ] References formatted correctly
- [ ] Cross-references work (`\ref{}`, `\label{}`)
- [ ] PDF file size < 10MB

---

## 11. Version Control

### File Naming

- Main paper: `paper.tex` (current version)
- Enhanced version: `paper_enhanced.tex`
- Submissions: `paper_submission_YYYY-MM-DD.tex`
- Revisions: `paper_revision_R1.tex`, `paper_revision_R2.tex`

### Change Tracking

Major changes should be logged in `/docs/writing_optimization_log.md` with:
- Date of change
- Nature of change
- Reason for change
- Author responsible

---

## 12. Contact and Updates

This style guide is a living document. Updates should be made when:
- New terminology is introduced
- Journal-specific requirements are identified
- Common errors are discovered during editing

**Last Updated**: 2026-02-13
**Version**: 1.0
**Maintainer**: Academic Writing Expert

---

## Appendix: Quick Reference Card

### Common LaTeX Snippets

```latex
% Citation
\citep{AuthorYear}
\citet{AuthorYear}

% Table
\begin{table}[H]
\centering
\caption{Caption}
\label{tab:name}
\begin{tabular}{lcc}
\toprule
A & B & C \\
\midrule
1 & 2 & 3 \\
\bottomrule
\end{tabular}
\end{table}

% Figure
\begin{figure}[H]
\centering
\includegraphics[width=0.8\linewidth]{path.png}
\caption{Caption}
\label{fig:name}
\end{figure}

% Equation
\begin{equation}
y = \beta x + \epsilon
\end{equation}

% Emphasis
\textbf{bold text}
\textit{italic text}
\textbf{``Sweet Spot''}
```

### Key Terminology at a Glance

| Term | Format | First Use |
|------|--------|-----------|
| DCI | Italic | Domestic Digital Capacity (DCI) |
| EDS | Italic | External Digital Specialization (EDS) |
| Sweet Spot | Bold | "Sweet Spot" |
| Causal Forest | Normal | Causal Forest DML |
| Model Ladder | Normal | Model Ladder |

---

*End of Style Guide*
