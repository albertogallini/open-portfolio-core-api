#!/usr/bin/env python3
"""
Regenerates Open-Portfolio-RiskModel.pdf with two new sections:
  §5.3  Security-Level Euler Risk Attribution  (get_portfolio_tree_risk)
  §7    Regime-Aware Portfolio Construction    (construct_portfolio)
Original §7 References renumbered to §8.
"""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Register Arial Unicode for full Unicode coverage (Greek, math symbols, etc.)
_AU  = '/Library/Fonts/Arial Unicode.ttf'
_AB  = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'
_ABI = '/System/Library/Fonts/Supplemental/Arial Bold Italic.ttf'
pdfmetrics.registerFont(TTFont('ArialUnicode',         _AU))
pdfmetrics.registerFont(TTFont('ArialUnicode-Bold',    _AB  if os.path.exists(_AB)  else _AU))
pdfmetrics.registerFont(TTFont('ArialUnicode-Italic',  _AU))
pdfmetrics.registerFont(TTFont('ArialUnicode-BoldItalic', _ABI if os.path.exists(_ABI) else _AU))
pdfmetrics.registerFontFamily('ArialUnicode',
    normal='ArialUnicode', bold='ArialUnicode-Bold',
    italic='ArialUnicode-Italic', boldItalic='ArialUnicode-BoldItalic')

PAGE_W, PAGE_H = A4
MARGIN = 2.2 * cm


# ─── Page footer callback ─────────────────────────────────────────────────────

def _draw_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#888888'))
    txt = f'Equity Factor Risk Model — Technical Reference | Page {doc.page}'
    canvas.drawCentredString(PAGE_W / 2, 1.2 * cm, txt)
    canvas.restoreState()


# ─── Styles ───────────────────────────────────────────────────────────────────

def build_styles():
    s = getSampleStyleSheet()
    kw = dict(
        normal=ParagraphStyle('Body', parent=s['Normal'],
            fontName='ArialUnicode', fontSize=10, leading=15,
            spaceAfter=6, alignment=TA_JUSTIFY),
        title=ParagraphStyle('DocTitle', parent=s['Title'],
            fontName='Helvetica-Bold', fontSize=20, leading=26,
            alignment=TA_CENTER, spaceAfter=4),
        subtitle=ParagraphStyle('Subtitle', parent=s['Normal'],
            fontName='ArialUnicode', fontSize=11, leading=15,
            alignment=TA_CENTER, spaceAfter=2,
            textColor=colors.HexColor('#555555')),
        tagline=ParagraphStyle('Tagline', parent=s['Normal'],
            fontName='ArialUnicode-Italic', fontSize=9, leading=13,
            alignment=TA_CENTER, spaceAfter=14,
            textColor=colors.HexColor('#777777')),
        h1=ParagraphStyle('H1', parent=s['Heading1'],
            fontName='Helvetica-Bold', fontSize=13, leading=18,
            spaceBefore=16, spaceAfter=6,
            textColor=colors.HexColor('#1a1a2e')),
        h2=ParagraphStyle('H2', parent=s['Heading2'],
            fontName='Helvetica-Bold', fontSize=11, leading=15,
            spaceBefore=12, spaceAfter=4,
            textColor=colors.HexColor('#2d2d5e')),
        h3=ParagraphStyle('H3', parent=s['Heading3'],
            fontName='Helvetica-BoldOblique', fontSize=10, leading=14,
            spaceBefore=10, spaceAfter=3,
            textColor=colors.HexColor('#3d3d6e')),
        eq=ParagraphStyle('Equation', parent=s['Normal'],
            fontName='ArialUnicode', fontSize=10, leading=15,
            leftIndent=28, spaceAfter=2, spaceBefore=4,
            backColor=colors.HexColor('#f5f5f5')),
        eq_num=ParagraphStyle('EqNum', parent=s['Normal'],
            fontName='Helvetica-Oblique', fontSize=9, leading=12,
            leftIndent=28, spaceAfter=8,
            textColor=colors.HexColor('#666666')),
        bullet=ParagraphStyle('Bullet', parent=s['Normal'],
            fontName='ArialUnicode', fontSize=10, leading=15,
            leftIndent=20, firstLineIndent=-10, spaceAfter=4),
        note=ParagraphStyle('Note', parent=s['Normal'],
            fontName='ArialUnicode-Italic', fontSize=9, leading=13,
            spaceBefore=4, spaceAfter=6,
            textColor=colors.HexColor('#555555')),
        footer=ParagraphStyle('Footer', parent=s['Normal'],
            fontName='ArialUnicode', fontSize=8, leading=10,
            alignment=TA_CENTER, textColor=colors.HexColor('#888888')),
    )
    return kw


# ─── Helpers ──────────────────────────────────────────────────────────────────

def fix_unicode(text):
    """Replace Unicode modifier-letter sub/superscripts with reportlab markup.
    Standard Greek letters (σ, Σ, ε, ξ …) stay as-is — ArialUnicode carries them.
    Compound sequences must come before their constituent single chars."""
    subs = [
        # Compound: dimension pairs joined by x
        ('ᵎ\xd7ᵏ', '<super>N\xd7K</super>'),
        ('ᵏ\xd7ᵏ', '<super>K\xd7K</super>'),
        # Compound: parenthesised superscripts
        ('⁽ᵀ⁺¹⁾', '<super>(T+1)</super>'),
        ('⁽ᵗ⁺¹⁾', '<super>(t+1)</super>'),
        ('⁽ᵀ⁾', '<super>(T)</super>'),
        ('⁽ᵗ⁾', '<super>(t)</super>'),
        ('⁽ᵏ*⁾', '<super>(k*)</super>'),
        ('⁽ᵏ⁾', '<super>(k)</super>'),
        # Other compound sequences
        ('ᵰa', 'N'),
        ('\xb9ʰʰ', '<super>1/h</super>'),
        ('ˢᵗᵃ', '<super>sta</super>'),
        ('⁻\xb9', '<super>-1</super>'),
        ('ₜ₊₁', '<sub>t+1</sub>'),
        # Single superscript modifier letters
        ('ᵀ', '<super>T</super>'),
        ('ᵏ', '<super>K</super>'),
        ('ᵎ', '<super>N</super>'),
        ('ᴰ', '<super>D</super>'),
        ('ᵂ', '<super>W</super>'),
        # Single subscript modifier letters / subscript digits
        ('ᵢ', '<sub>i</sub>'),
        ('ₚ', '<sub>p</sub>'),
        ('ₖ', '<sub>k</sub>'),
        ('ₜ', '<sub>t</sub>'),
        ('ₙ', '<sub>n</sub>'),
        ('₁', '<sub>1</sub>'),
        ('₂', '<sub>2</sub>'),
        ('₀', '<sub>0</sub>'),
        ('₌', '='),
        # Remaining catch-all parenthetical superscripts
        ('⁽', '<super>(</super>'),
        ('⁾', '<super>)</super>'),
        # Misc
        ('\xbd', '1/2'),
    ]
    for old, new in subs:
        text = text.replace(old, new)
    return text


def p(text, st):
    return Paragraph(fix_unicode(text), st)

def sp(h=6):
    return Spacer(1, h)

def hr():
    return HRFlowable(width='100%', thickness=0.5,
                      color=colors.HexColor('#cccccc'), spaceAfter=8)

def sec(num, title, s):
    return p(f'{num}. {title}', s['h1'])

def sub(num, title, s):
    return p(f'{num} {title}', s['h2'])

def sub3(num, title, s):
    return p(f'{num} {title}', s['h3'])

def formula(text, label, s):
    return [p(text, s['eq']), p(f'({label})', s['eq_num'])]

def bul(text, s):
    return p(f'•  {text}', s['bullet'])

def tbl(data, widths, has_header=True):
    t = Table(data, colWidths=widths)
    style = [
        ('FONTNAME',    (0, 0), (-1, -1), 'ArialUnicode'),
        ('FONTSIZE',    (0, 0), (-1, -1), 9),
        ('LEADING',     (0, 0), (-1, -1), 13),
        ('VALIGN',      (0, 0), (-1, -1), 'TOP'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING',(0, 0), (-1, -1), 6),
        ('TOPPADDING',  (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING',(0,0), (-1, -1), 4),
        ('GRID',        (0, 0), (-1, -1), 0.3, colors.HexColor('#cccccc')),
        ('ROWBACKGROUNDS', (0, 1 if has_header else 0), (-1, -1),
         [colors.HexColor('#f9f9f9'), colors.white]),
    ]
    if has_header:
        style += [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
            ('TEXTCOLOR',  (0, 0), (-1, 0), colors.white),
            ('FONTNAME',   (0, 0), (-1, 0), 'ArialUnicode-Bold'),
        ]
    t.setStyle(TableStyle(style))
    return t


# ─── Document content ─────────────────────────────────────────────────────────

def story(s):
    S = []

    # Title block
    S += [
        sp(20),
        p('Equity Factor Risk Model', s['title']),
        p('Technical Reference for Quantitative Practitioners', s['subtitle']),
        sp(4),
        p('Barra-Style Cross-Sectional Model · Arbitrary Equity Universe '
          '· EWMA Covariance Calibration', s['tagline']),
        p('Version 2.1 | Daily Calibration | Universe: Configurable', s['tagline']),
        hr(),
        sp(8),
    ]

    # ── §1 Abstract ──────────────────────────────────────────────────────────
    S += [
        sec('1', 'Abstract', s),
        p('This document describes the specification, mathematical framework, and '
          'implementation of a Barra-style cross-sectional factor risk model for any equity '
          'universe. The model follows the Fundamental Factor Model paradigm introduced by '
          'Rosenberg (1974) and later formalised in the MSCI Barra family of models. Each '
          'trading day, stock returns are regressed on a pre-specified set of K factor '
          'exposures across an N-stock universe. The resulting factor returns and idiosyncratic '
          'residuals are used to maintain Exponentially Weighted Moving Average (EWMA) '
          'covariance matrices. The calibrated model delivers three primary outputs for every '
          'trading day: the full factor covariance matrix F [K × K], the idiosyncratic '
          'variance vector d [N], and per-stock or per-portfolio risk decompositions under the '
          'identity Σ = XFXᵀ + D.', s['normal']),
        sp(4),
        p('<b>Version 2.0</b> adds two capabilities: (i) a security-level Euler '
          'risk-contribution decomposition (§5.3) that attributes portfolio volatility '
          'exactly across holdings, and (ii) a regime-aware portfolio construction engine '
          '(§7) that conditions the mean-variance objective on the active market regime '
          'identified by the Wasserstein-HMM model.', s['normal']),
        sp(4),
        p('<b>Version 2.1</b> corrects a set of point-in-time and estimation-robustness '
          'issues found in an internal review of the Version 2.0 pipeline: (i) cross-sectional '
          'exposures used in the daily regression are now built as of t−1 rather than t, '
          'removing a look-ahead leak of the regressand into several price-based factors '
          '(§6.1); (ii) fundamentals are filtered by a configurable reporting lag rather than '
          'quarter-end date, and cached raw per ticker so the point-in-time filter no longer '
          'depends on call order (§6.1, §6.3); (iii) the regime model’s conditional factor '
          'means are now estimated predictively rather than as an in-sample average of a '
          'regime’s own labelled days, and reported confidence is the one-step-ahead '
          'predicted regime probability rather than the filtered probability at T (§7.1–'
          '7.2); (iv) the regime HMM’s covariance is regularised (diagonal by default) and '
          'its inputs are causally standardised, with a guardrail rejecting fits whose '
          'estimation window cannot support the number of free parameters (§7.1); (v) the '
          'portfolio construction objective is scaled to the expected holding period rather '
          'than a single day, and the turnover penalty is solved as a smooth QP that raises '
          'on solver failure instead of silently returning equal weights (§7.3).', s['normal']),
    ]

    # ── §2 Model Structure ───────────────────────────────────────────────────
    S += [
        sec('2', 'Model Structure', s),
        sub('2.1', 'The Fundamental Factor Model', s),
        p('At date t, let r ∈ ℝᵎ be the vector of stock excess returns and '
          'X ∈ ℝᵎ×ᵏ be the exposure matrix, with each row xᵢ '
          'giving stock i&#39;s loadings on K factors. The model assumes:', s['normal']),
    ]
    S += formula('r = X f + ε', '1', s)
    S += [
        p('where f ∈ ℝᵏ is the vector of factor returns (estimated via '
          'regression) and ε ∈ ℝᵎ is the vector of idiosyncratic returns, '
          'assumed uncorrelated across stocks: E[εεᵀ] = D, where D is diagonal.',
          s['normal']),
        p('Under this structure, the full N × N covariance matrix of returns is:',
          s['normal']),
    ]
    S += formula('Σ = X F Xᵀ + D', '2', s)
    S += [
        p('where F = E[ffᵀ] ∈ ℝᵏ×ᵏ is the factor covariance '
          'matrix. Because K ≪ N (typically K ≈ 20, N ≈ 500), the factor '
          'structure reduces estimation burden from O(N²) free parameters to O(K²) + '
          'O(N), a compression of roughly 1,000-fold for a 500-stock universe (indicative).',
          s['normal']),
    ]

    S += [
        sub('2.2', 'Cross-Sectional Weighted Least Squares', s),
        p('Factor returns are not pre-specified (as in Fama–French) but estimated daily '
          'via cross-sectional Weighted Least Squares (WLS). Stock i receives weight '
          'wᵢ = √(Mᵢ), where Mᵢ is market capitalisation. This scheme '
          'down-weights micro-cap names whose returns are noisier and privileges large-cap '
          'stocks whose prices are more informationally efficient.', s['normal']),
        p('The WLS estimator solves the weighted problem: '
          'min ∑ᵢ wᵢ (rᵢ − xᵢᵀ f)², '
          'yielding the closed-form solution:', s['normal']),
    ]
    S += formula('f̂ = (Xᵀ W X)⁻¹ Xᵀ W r', '3', s)
    S += [
        p('where W = diag(w₁, …, wₙ). The implementation transforms to '
          'standard OLS via the substitution X̃ = W½ X, r̃ = W½ r, '
          'and applies numpy.linalg.lstsq for numerical robustness. Idiosyncratic returns '
          'are then recovered as ε̂ᵢ = rᵢ − xᵢᵀ f̂.',
          s['normal']),
        p('Factor significance is evaluated daily using approximate homoskedastic t-statistics:',
          s['normal']),
    ]
    S += formula('t̂ₖ = f̂ₖ / √{ [(Xᵀ W X)⁻¹]ₖₖ · MSE }', '4', s)
    S += [
        p('where MSE = (1/N) ∑ᵢ ε̂ᵢ². A daily t-statistic above 2.0 '
          '(in absolute value) is considered significant. The fraction of days on which '
          '|t̂ₖ| > 2 provides a practical persistence metric for each factor.',
          s['normal']),
    ]

    # ── §3 Factor Catalogue ──────────────────────────────────────────────────
    S += [
        sec('3', 'Factor Catalogue', s),
        p('The model contains K = 7 style factors, 2 technical factors, and sector dummies '
          '(one-hot per industry sector with one reference category dropped to avoid perfect '
          'multicollinearity). All continuous factors are cross-sectionally winsorised at '
          '±3σ and then standardised to zero mean, unit variance before regression.',
          s['normal']),
        sp(6),
        tbl([
            ['Factor', 'Group', 'Definition', 'Construction'],
            ['value_ep',    'Style',    'Earnings Yield',   'EP = TTM Net Income / Mkt Cap'],
            ['value_bp',    'Style',    'Book Yield',       'BP = Book Equity / Mkt Cap'],
            ['momentum',    'Style',    'Price Momentum',   'r(t−252, t−21)'],
            ['quality_roe', 'Style',    'Return on Equity', 'ROE = TTM NI / Book Eq.'],
            ['quality_gm',  'Style',    'Gross Margin',     'GM = TTM Gross Profit / Revenue'],
            ['size',        'Style',    'Log Market Cap',   'ln(P × shares outstanding)'],
            ['low_vol',     'Style',    'Low Volatility',   '−σ(252d realised vol)'],
            ['short_rev',   'Technical','Short Reversal',   '−r(t−21, t)'],
            ['liquidity',   'Technical','Log Notional Vol', 'ln(avg 20d notional volume)'],
            ['sector_*',    'Sector',   'Industry Dummies', 'One-hot per sector (ref dropped)'],
        ], [3.2*cm, 2.4*cm, 4*cm, 6.4*cm]),
        sp(6),
        p('<i>Note:</i> Momentum skips the most recent month (t−21 to t) to avoid '
          'contamination from microstructure reversal. Low-vol and short-reversal are '
          'sign-flipped so that positive loadings uniformly map to desirable characteristics '
          '(lower risk, less reversal drag).', s['note']),
    ]

    S += [
        sub('3.1', 'Normalisation Protocol', s),
        p('For each continuous factor column fₖ across the N stocks on date t:', s['normal']),
        bul('<b>Step 1 — Winsorise:</b> clip to [µₖ − 3σₖ, '
            'µₖ + 3σₖ] where µₖ, σₖ are the '
            'cross-sectional mean and standard deviation.', s),
        bul('<b>Step 2 — Standardise:</b> recompute µ̃ₖ, σ̃ₖ '
            'on the clipped series and z-score: z̃ᵢₖ = (f̃ᵢₖ '
            '− µ̃ₖ) / σ̃ₖ.', s),
        p('Stocks missing the fundamental input for factor k receive NaN for that column. '
          'The WLS estimator drops any stock with at least one NaN across its K exposures, '
          'so thin-data stocks are automatically excluded without biasing the regression.',
          s['normal']),
    ]

    # ── §4 Covariance Estimation ─────────────────────────────────────────────
    S += [
        sec('4', 'Covariance Estimation', s),
        sub('4.1', 'EWMA Dynamics', s),
        p('Both the factor covariance matrix and the idiosyncratic variances are updated online '
          'using Exponentially Weighted Moving Averages. The EWMA scheme avoids storing a full '
          'return history while placing geometrically declining weight on older observations.',
          s['normal']),
        p('Let λ = 0.5¹ʰʰ where h is the chosen half-life in trading days. '
          'The recursion for the factor covariance matrix is:', s['normal']),
    ]
    S += formula('F̂ₜ = λ · F̂ₜ₋₁ + (1−λ) · f̂ₜ f̂ₜᵀ', '5', s)
    S += [p('Analogously, for each stock i, the idiosyncratic variance estimate follows:', s['normal'])]
    S += formula('d̂ᵢ,ₜ = λᴰ · d̂ᵢ,ₜ₋₁ + (1−λᴰ) · ε̂ᵢ,ₜ²', '6', s)
    S += [
        p('The two half-lives are deliberately set at different horizons: factor covariance is '
          'updated at a 63-day half-life (approximately 3 calendar months) to capture persistent '
          'systematic risk regimes, while idiosyncratic variance uses a 21-day half-life '
          '(approximately 1 month) to respond rapidly to event-driven stock-specific volatility.',
          s['normal']),
        sp(4),
        tbl([
            ['Matrix', 'Half-life', 'Decay factor λ', 'Approx. window'],
            ['Factor covariance F', '63 trading days', 'λ = 0.989', '~3 months'],
            ['Idiosyncratic var D', '21 trading days', 'λ = 0.967', '~1 month'],
        ], [5*cm, 3.5*cm, 3.5*cm, 4*cm]),
        sp(6),
        p('Because λ is strictly between 0 and 1, the EWMA estimator is always positive '
          'semi-definite: each update is a convex combination of a PSD matrix (the prior '
          'F̂ₜ₋₁) and a rank-1 PSD matrix (the outer product '
          'f̂ₜ f̂ₜᵀ). The full covariance Σ = X F̂ Xᵀ + '
          'diag(d̂) is therefore always positive definite as long as d̂ᵢ > 0 '
          '∀ i, which holds by construction.', s['normal']),
    ]

    S += [
        sub('4.2', 'Initialisation and Burn-in', s),
        p('On the first calibration date, both F̂ and each d̂ᵢ are initialised '
          'with the contemporaneous outer-product and squared residual respectively. The model '
          'then requires a burn-in period before estimates are reliable. In practice, no risk '
          'output should be consumed until at least one half-life of the slower EWMA (63 trading '
          'days ≈ 3 months) has been accumulated. The data loader automatically pre-fetches '
          'an additional 378 trading days of history (1.5 calendar years) before the user&#39;s '
          'analysis start date to fully warm the price-based factors.', s['normal']),
    ]

    # ── §5 Risk Decomposition ────────────────────────────────────────────────
    S += [
        sec('5', 'Risk Decomposition', s),
        sub('5.1', 'Stock-Level Decomposition', s),
        p('For a single stock i with exposure vector xᵢ ∈ ℝᵏ, the annualised '
          'total variance (scaling daily variance by 252) decomposes as:', s['normal']),
    ]
    S += formula('σᵢ² = xᵢᵀ F̂ xᵢ + d̂ᵢ\n'
                 '       = σᵢ,factor² + σᵢ,idio²', '7', s)
    S += [
        p('The factor share is defined as ρᵢ = σᵢ,factor² / '
          'σᵢ², providing an intuitive measure of systematic vs stock-specific '
          'risk. A name with ρᵢ close to 1 behaves like a pure beta play; ρᵢ '
          'near 0 implies the stock is largely driven by idiosyncratic events (e.g. biotech '
          'binary events, litigation).', s['normal']),
    ]

    S += [
        sub('5.2', 'Portfolio Risk and Factor Attribution', s),
        p('For a portfolio with weight vector w ∈ ℝᵎ (with ∑ᵢ wᵢ = 1), '
          'the portfolio variance is:', s['normal']),
    ]
    S += formula('σₚ² = wᵀ Σ w = wᵀ X F̂ Xᵀ w + wᵀ D̂ w', '8', s)
    S += [p('Define the portfolio factor exposure vector ξ = Xᵀ w ∈ ℝᵏ. Then:', s['normal'])]
    S += formula('σₚ,factor² = ξᵀ F̂ ξ ,    σₚ,idio² = ∑ᵢ wᵢ² d̂ᵢ', '9', s)
    S += [p('The marginal contribution of factor k to total portfolio factor variance is:', s['normal'])]
    S += formula('MCVₖ = ξₖ · (F̂ ξ)ₖ', '10', s)
    S += [
        p('The vector (MCVₖ)ₖ₌₁…ᵏ sums exactly to σₚ,factor², '
          'enabling a clean attribution of factor risk to individual model factors. This is '
          'surfaced in the UI as a bar chart of factor variance contributions, expressed both in '
          'absolute annual variance units and as a percentage of total portfolio variance.',
          s['normal']),
    ]

    # ── §5.3 Euler Risk Attribution (NEW) ────────────────────────────────────
    S += [
        KeepTogether([
            sub('5.3', 'Security-Level Euler Risk Attribution', s),
            p('While §5.2 computes a scalar portfolio variance, risk management also '
              'requires knowing <i>which positions drive that variance</i>. The '
              '<b>Euler decomposition</b> provides an exact, additive per-security breakdown '
              'that aggregates to the total portfolio volatility without any approximation.',
              s['normal']),
        ]),
        sub3('5.3.1', 'Mathematical Foundation', s),
        p('<b>Euler&#39;s theorem.</b> For any positively homogeneous risk measure '
          'ℛ(w) of degree 1:', s['normal']),
    ]
    S += formula('ℛ(w) = ∑ᵢ wᵢ ∂ℛ / ∂wᵢ', '11', s)
    S += [
        p('Applied to portfolio volatility σₚ = √(wᵀ Σ w), '
          'the marginal risk contribution of security i is:', s['normal']),
    ]
    S += formula('∂σₚ / ∂wᵢ = (Σ w)ᵢ / σₚ', '12', s)
    S += [p('The <b>Euler risk contribution</b> (in vol units) of security i is:', s['normal'])]
    S += formula('RCᵢ = wᵢ · (Σ w)ᵢ / σₚ ,    and    ∑ᵢ RCᵢ = σₚ  (exact)', '13', s)

    S += [
        sub3('5.3.2', 'Efficient Computation of Σ w', s),
        p('The factor model structure allows Σ w to be computed without ever '
          'materialising the N × N matrix, at cost O(NK + K²):', s['normal']),
    ]
    S += formula('(Σ w)ᵢ = xᵢᵀ F (Ωᵀ w)   +   d̂ᵢ · wᵢ\n'
                 '           ———————   ——————\n'
                 '              factor part        idio part', '14', s)
    S += [
        p('Evaluation proceeds in two steps: (1) compute ξ = Xᵀ w [K-vector, O(NK)]; '
          '(2) compute X (Fξ) [N-vector, O(NK + K²)]; (3) add d̂ ⊙ w '
          'element-wise [O(N)].', s['normal']),
    ]

    S += [
        sub3('5.3.3', 'Variance Shares and Annualised Contributions', s),
        p('The fractional variance contribution (sums to 100):', s['normal']),
    ]
    S += formula('RCᵢ% = wᵢ · (Σ w)ᵢ / σₚ² × 100 ,    ∑ᵢ RCᵢ% = 100', '15', s)
    S += [p('The annualised volatility contribution in percentage points (sums to total_vol):', s['normal'])]
    S += formula('RCi_vol (%) = wi · (Σ w)i / σp × √252 × 100', '16', s)

    S += [
        sub3('5.3.4', 'Standalone vs. Contribution Risk', s),
        p('The standalone vol σᵢˢᵗᵃ (equation (7)) and the Euler '
          'contribution RCi_vol (equation (16)) satisfy the inequality:', s['normal']),
        p('σᵢˢᵗᵃ ≥ RCi_vol  (by Cauchy-Schwarz), '
          'with equality iff the portfolio has exactly one name.', s['normal']),
        p('The ratio RCi_vol / σᵢˢᵗᵃ captures how much '
          "of a name&#39;s standalone risk is diversified away. A name with a high ratio "
          'contributes more risk than diversification removes; a low ratio signals that the '
          'position is largely hedged by the rest of the portfolio.', s['normal']),
    ]

    S += [
        sub3('5.3.5', 'API Output: get_portfolio_tree_risk', s),
        p('The API returns a DataFrame with one row per holding and a <tt>portfolio</tt> '
          'summary row. The portfolio row carries the same columns as <tt>get_portfolio_risk</tt> '
          '(including VaR fields). Security-row columns in the portfolio row equal the portfolio '
          'aggregates; portfolio-row columns in security rows are NaN.', s['normal']),
        sp(4),
        tbl([
            ['Column', 'Security rows', 'Portfolio row'],
            ['weight',       'Portfolio weight (0–100)',          '100'],
            ['total_vol',    'Standalone annualised vol (%)',          'Portfolio vol (%)'],
            ['factor_vol',   'Factor component of standalone vol',    'Portfolio factor vol'],
            ['idio_vol',     'Idio component of standalone vol',      'Portfolio idio vol'],
            ['factor_share', 'Factor share of standalone variance',   'Portfolio factor share'],
            ['RC_vol',       'Eq.(16): Euler vol contribution, sums to total_vol',  '= total_vol'],
            ['RC_pct',       'Eq.(15): sums to 100',                  '100.0'],
            ['total_var',    '—',                                 'Annualised portfolio variance'],
            ['sigma_daily',  '—',                                 'Daily 1-sigma (%)'],
            ['sigma_horizon','—',                                 'Horizon-scaled sigma (%)'],
            ['VaR_pct',      '—',                                 'Parametric VaR (% of NAV)'],
            ['VaR_value',    '—',                                 'VaR in currency (optional)'],
        ], [3.5*cm, 6.5*cm, 6*cm]),
        sp(8),
    ]

    # ── §6 Implementation Overview ───────────────────────────────────────────
    S += [
        sec('6', 'Implementation Overview', s),
        sub('6.1', 'Daily Calibration Pipeline', s),
        p('The calibration pipeline executes the following steps at market close each '
          'trading day:', s['normal']),
        bul('<b>Data ingestion:</b> adjusted close prices and volumes for the full universe '
            'are fetched from the bulk panel (pre-loaded at startup). Quarterly income '
            'statement, cash-flow, and balance sheet data are fetched per-ticker and their '
            'raw (unfiltered) frames cached by ticker alone; the point-in-time filter '
            '(period-end date ≤ target_date − report_lag_days, default 45 days) is '
            're-applied on every call so the result never depends on call order within a '
            'quarter.', s),
        bul('<b>Return computation:</b> one-day log returns rᵢ,ₜ = '
            'ln(Pᵢ,ₜ / Pᵢ,ₜ₋₁) are computed. Observations outside '
            '[−50%, +50%] are dropped to filter corporate actions and data errors.', s),
        bul('<b>Exposure matrix:</b> FactorBuilder computes the N × K exposure matrix used in '
            'the regression <i>as of t−1</i>, not t: because price-based factors '
            '(short_rev, size, value_ep/bp, liquidity, low_vol) read history up to and '
            'including their target date, building them at t would place '
            'rᵢ,ₜ itself inside the regressors. A second exposure matrix is built '
            'as of the latest available date for forecasting / risk-decomposition queries, '
            'where no such constraint applies. Stocks with fewer than 252 trading days of '
            'price history are dropped. All continuous factors are winsorised and z-scored '
            'cross-sectionally. Sector dummies are appended.', s),
        bul('<b>WLS regression:</b> equation (3) is solved via transformed OLS. t-statistics '
            '(4) are computed. The regression is skipped if fewer than 50 complete-observation '
            'stocks are available.', s),
        bul('<b>EWMA update:</b> equations (5) and (6) are applied to update F̂ and each '
            'd̂ᵢ. New stocks entering the sample receive initialised idiosyncratic '
            'variance equal to their first squared residual.', s),
        bul('<b>Persistence:</b> RegressionResult objects (factor returns, residuals, exposures, '
            'R²) are stored in memory. The EWMA matrices update in place for efficiency.', s),
    ]

    S += [
        sub('6.2', 'Computational Complexity', s),
        p('The dominant cost per day is the exposure matrix construction, which involves one '
          'API call per ticker for the fundamentals if not cached. After the warm-up period, '
          'the vast majority of fundamental calls hit the in-memory quarterly cache, reducing '
          'marginal cost to approximately O(N·K) for the matrix algebra. The WLS solve '
          'via lstsq is O(N·K²), and the EWMA update is O(K²) — both '
          'negligible at K ≈ 20, N ≈ 500.', s['normal']),
    ]

    S += [
        sub('6.3', 'Known Limitations and Extensions', s),
        bul('<b>Estimation risk in F̂:</b> with K ≈ 20 and a 63-day effective window, '
            'the factor covariance matrix is susceptible to Stein-type shrinkage improvements '
            '(Ledoit–Wolf, Oracle Approximating Shrinkage). A shrinkage target of the '
            'scaled identity is a recommended near-term extension.', s),
        bul('<b>Reporting-lag approximation:</b> the report_lag_days filter (default 45 days) '
            'approximates publication delay from period-end date, since the data source '
            'exposes period-end dates rather than actual filing dates; the true 10-Q/10-K '
            'filing date would be a more precise (but currently unavailable) point-in-time '
            'boundary. Shares outstanding is also a current, not historical, figure applied '
            'to every historical date, which slightly skews size and market-cap-based ratios '
            'for names with material buybacks/issuance in the lookback window.', s),
        bul('<b>Factor sparsity:</b> stocks missing all fundamental data (recent IPOs) are '
            'excluded from the regression. A missing-data imputation scheme — for instance, '
            'cross-sectional median within sector — would increase the effective universe.', s),
        bul('<b>Macro overlays:</b> the current model is pure fundamental/technical. Adding '
            'observable macro factor returns (VIX changes, credit spread changes, yield curve '
            'slope) as pre-specified factors would capture systematic macro risk without '
            'estimation error in the exposure matrix.', s),
        bul('<b>Non-linear risk:</b> the model is purely linear. Portfolios with options or '
            'non-linear payoffs require a Monte Carlo or delta-gamma overlay to convert the '
            'linear factor model into a full P&L distribution.', s),
    ]

    # ── §7 Regime-Aware Portfolio Construction (NEW) ─────────────────────────
    S += [
        sec('7', 'Regime-Aware Portfolio Construction', s),
        p('The <tt>construct_portfolio</tt> API combines the factor risk model (§§2'
          '–5) with a Wasserstein-HMM regime model to build a long-only, '
          'transaction-cost-aware portfolio conditioned on the active market regime. '
          'This section describes the mathematical framework of the construction engine.',
          s['normal']),
    ]

    S += [
        sub('7.1', 'Prerequisites: Calibration Pipeline', s),
        p('Two calibration steps must be executed in order before calling '
          '<tt>construct_portfolio</tt>:', s['normal']),
        bul('<b><tt>risk_calibration</tt></b> — fits the factor model (§§2–4) '
            'on the ticker universe derived from portfolio holdings and serialises the fitted '
            'calibrator to <tt>risk_model.pkl</tt>.', s),
        bul('<b><tt>regime_calibration</tt></b> — fits a Wasserstein-HMM on the factor-return '
            'history embedded in <tt>risk_model.pkl</tt>. The HMM uses a strictly causal rolling '
            'window with predictive model-order selection (Wasserstein identity tracking ensures '
            'regime labels are stable across refits). Inputs are z-scored by each factor’s '
            'own trailing (causal, EWM) volatility before fitting, since factor volatilities can '
            'differ 2× or more and both the Gaussian likelihood and the Wasserstein template '
            'matching are scale-sensitive. State covariance is diagonal by default '
            '(configurable to shrunk-toward-diagonal or full), regularised by a diagonal '
            'loading proportional to the window’s own factor variances rather than a fixed '
            'constant. The calibrator rejects a window whose estimation_window − '
            'order_selection_holdout is too small relative to the HMM’s free-parameter count '
            'at max_regimes, raising a readable error rather than fitting a near-singular model. '
            'The fitted calibrator is serialised to <tt>regime_model.pkl</tt>.', s),
        p('Let π⁽ᵀ⁾ = (π₁⁽ᵀ⁾, …, '
          'πᵏ⁽ᵀ⁾)ᵀ be the K-vector of filtered regime probabilities '
          'at the most recent observation T, and A the K×K persistent-identity transition '
          'matrix. The active regime is '
          'k* = argmaxₖ πₖ⁽ᵀ⁾.', s['normal']),
        p('The reported <i>regime confidence</i> and the expected-return forecast (§7.2) do '
          'not use π⁽ᵀ⁾ directly. Both instead use the one-step-ahead <b>predicted</b> '
          'regime distribution:', s['normal']),
    ]
    S += formula('π⁽ᵀ⁺¹⁾ = π⁽ᵀ⁾ A', '17', s)
    S += [
        p('π⁽ᵀ⁾ describes where the model believes the process currently IS; '
          'π⁽ᵀ⁺¹⁾ is the belief about where it will be next, which is the '
          'relevant quantity for anything forward-looking — the expected return used in '
          'portfolio construction, and the confidence figure reported alongside it.',
          s['normal']),
    ]

    S += [
        sub('7.2', 'Regime-Conditional Expected Returns', s),
        p('For each security i in the investment universe, the regime-conditional expected '
          'return is obtained by projecting its factor exposures onto a factor mean vector λ:',
          s['normal']),
    ]
    S += formula('μᵢ = xᵢᵀ λ', '18', s)
    S += [
        p('λ ∈ ℝᵏ is <b>not</b> the in-sample average of factor returns on days '
          'labelled k* — averaging a regime’s own labelled days is biased by construction, '
          'since the labels themselves come from fitting on those same returns. Instead, for '
          'each persistent regime id, its mean/covariance are estimated by weighting the '
          '<i>next-day</i> factor return by the causal filtered belief at t:', s['normal']),
    ]
    S += formula('λ⁽ᵏ⁾ = ∑ₜ πₖ⁽ᵗ⁾ fₜ₊₁  /  ∑ₜ πₖ⁽ᵗ⁾', '19', s)
    S += [
        p('By default (no regime pinned by the caller), λ is the one-step-ahead <b>forecast</b>: '
          'the blend of every tracked regime’s λ⁽ᵏ⁾ weighted by the predicted next-day '
          'belief π⁽ᵀ⁺¹⁾ (equation 17), λ = ∑ₖ πₖ⁽ᵗ⁺¹⁾ λ⁽ᵏ⁾, rather than a hard pick of '
          'k*. Pinning a specific regime id instead returns that regime’s own λ⁽ᵏ⁾, for '
          'scenario analysis.', s['normal']),
        p('The regime model only speaks to the continuous style/technical factors. Sector-'
          'dummy factors — which act as the market/beta term — receive a flat 0 prior rather '
          'than their unconditional historical mean, since handing every stock a 30–40% '
          'annualised bull-market baseline as an "expected return" is extrapolation, not a '
          'regime call. The regime-sensitive factors are shrunk toward their own unconditional '
          'mean λ_uncond by a configurable α ∈ [0,1] rather than trusted outright:',
          s['normal']),
    ]
    S += formula('λ_final = λ_uncond + α · (λ − λ_uncond)', '20', s)

    S += [
        sub('7.3', 'Mean-Variance Optimisation with Transaction Costs', s),
        p('Given the universe ᵰa and expected holding period h (the active regime’s '
          'expected dwell time, 1 / (1 − p_k*,k*)), the optimiser solves a long-only '
          'mean-variance problem with an ℓ₁ turnover penalty:', s['normal']),
    ]
    S += formula('w* = argmaxᵂ [ h·μᵀ w  −  (γ/2)·h·wᵀ Σ w  −  cᵀ |w − w₀| ]', '21', s)
    S += [
        p('subject to: w ≥ 0 (long-only),  ∑ᵢ wᵢ = 1 (fully invested),  '
          'wᵢ ≤ w_max ∀ i (optional name cap)', s['normal']),
        p('where γ is the risk-aversion coefficient, c is the per-unit transaction cost '
          'vector (bps, applied symmetrically to buys and sells), and w₀ are the current '
          'holdings weights. μ and Σ are scaled by h because both are linear in time and the '
          'transaction cost term is a one-off charge: comparing it against a single day’s '
          'expected return, rather than the return actually expected over the holding period, '
          'understates the return side of the trade-off for anything other than a 1-day hold. '
          'The ℓ₁ turnover penalty is made smooth by splitting the trade w − w₀ = b − s into '
          'buy/sell auxiliary variables b, s ≥ 0 — at any optimum of this convex objective the '
          'solver never sets both bᵢ and sᵢ positive simultaneously, so bᵢ + sᵢ = '
          '|wᵢ − w₀,ᵢ| exactly — yielding a smooth QP solved with SLSQP. If the '
          'solver does not report success, the API raises rather than silently falling back '
          'to an equal-weight portfolio.', s['normal']),
    ]

    S += [
        sub('7.4', 'Cardinality Pre-Selection', s),
        p('When <tt>target_n</tt> is specified, names are pre-ranked by their regime-adjusted '
          'information ratio:', s['normal']),
    ]
    S += formula('IRᵢ = μᵢ / σᵢˢᵗᵃ', '22', s)
    S += [
        p('and the top <tt>target_n</tt> names are passed to the QP. This greedy pre-selection '
          'is O(N log N) and typically produces near-optimal cardinality while keeping the '
          'optimisation tractable.', s['normal']),
    ]

    S += [
        sub('7.5', 'Output Schema', s),
        sp(4),
        tbl([
            ['Column', 'Description'],
            ['weight',                'Optimal target weight (0–1 scale)'],
            ['prev_weight',           'Pre-trade weight from universe holdings (0 if from cash)'],
            ['trade',                 'Suggested rebalancing trade: weight − prev_weight'],
            ['expected_return_regime','Regime-conditional expected return mu_i (equation 18)'],
            ['regime_id',             'Displayed regime index (current regime, or a caller-pinned id)'],
            ['regime_confidence',     'Predicted next-day probability pi(T+1) of that regime (equation 17), not the filtered probability at T'],
            ['expected_hold_days',    'Expected regime dwell time: 1 / (1 - p_k*,k*)'],
        ], [5*cm, 12*cm]),
        sp(8),
    ]

    S += [
        sub('7.6', 'Closed Loop: Construction and Attribution', s),
        p('The covariance matrix Σ used in equation (21) is the same EWMA factor-model '
          'covariance as in §4: Σ = X F̂ Xᵀ + D̂. The Euler '
          'risk-contribution API (<tt>get_portfolio_tree_risk</tt>, §5.3) can therefore '
          'be applied immediately to the constructed portfolio w* to compute the security-level '
          'risk budget implied by the regime-optimal allocation. This provides a closed loop '
          'between construction and attribution: the same factor model governs both the '
          'optimisation objective and the ex-ante risk reporting.', s['normal']),
    ]

    # ── §8 References ────────────────────────────────────────────────────────
    S += [
        sec('8', 'References', s),
        p('Rosenberg, B. (1974). Extra-Market Components of Covariance in Security Returns. '
          '<i>Journal of Financial and Quantitative Analysis</i>, 9(2), 263–274.', s['normal']),
        p('Grinold, R. &amp; Kahn, R. (2000). <i>Active Portfolio Management</i> (2nd ed.). '
          'McGraw-Hill.', s['normal']),
        p('MSCI Barra (2011). Barra Fundamental Factor Models — Empirical Notes. '
          'MSCI Research.', s['normal']),
        p('Fama, E. &amp; MacBeth, J. (1973). Risk, Return, and Equilibrium: Empirical Tests. '
          '<i>Journal of Political Economy</i>, 81(3), 607–636.', s['normal']),
        p('Ledoit, O. &amp; Wolf, M. (2004). A well-conditioned estimator for large-dimensional '
          'covariance matrices. <i>Journal of Multivariate Analysis</i>, 88(2), 365–411.',
          s['normal']),
        p('RiskMetrics Group (1996). <i>RiskMetrics Technical Document</i> (4th ed.). '
          'J.P. Morgan/Reuters.', s['normal']),
        p('Hamilton, J.D. (1989). A new approach to the economic analysis of nonstationary '
          'time series and the business cycle. <i>Econometrica</i>, 57(2), 357–384.',
          s['normal']),
        p('Villani, C. (2008). <i>Optimal Transport: Old and New</i>. Springer.', s['normal']),
        sp(20),
        hr(),
        p('© Internal. This document is for qualified professional use only and does '
          'not constitute investment advice.', s['footer']),
    ]

    return S


# ─── Build and write ──────────────────────────────────────────────────────────

def main():
    out_paths = [
        '/Users/albertogallini/Openportfoliows/public/assets/Open-Portfolio-RiskModel.pdf',
        '/Users/albertogallini/Openportfoliows/src/assets/Open-Portfolio-RiskModel.pdf',
        '/Users/albertogallini/pythonprj/Openportfoliows/public/assets/Open-Portfolio-RiskModel.pdf',
        '/Users/albertogallini/pythonprj/Openportfoliows/src/assets/Open-Portfolio-RiskModel.pdf',
    ]

    st = build_styles()
    content = story(st)

    for path in out_paths:
        if not os.path.isdir(os.path.dirname(path)):
            print(f'Skipping (directory missing): {path}')
            continue
        doc = SimpleDocTemplate(
            path,
            pagesize=A4,
            leftMargin=MARGIN, rightMargin=MARGIN,
            topMargin=MARGIN, bottomMargin=2.8 * cm,
        )
        doc.build(content, onFirstPage=_draw_footer, onLaterPages=_draw_footer)
        print(f'Written: {path}')


if __name__ == '__main__':
    main()
