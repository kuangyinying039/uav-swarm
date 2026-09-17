#!/usr/bin/env python3
"""Replace EQ* placeholders with native Office Math (OMML / MathType-compatible)."""
from __future__ import annotations

import sys
from lxml import etree
from pptx import Presentation

M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"


etree.register_namespace("m", M)


def me(tag):
    return f"{{{M}}}{tag}"


def ae(tag):
    return f"{{{A}}}{tag}"


def r(text, sty="i"):
    node = etree.Element(me("r"))
    rpr = etree.SubElement(node, me("rPr"))
    sty_el = etree.SubElement(rpr, me("sty"))
    sty_el.set(f"{{{M}}}val", sty)
    t = etree.SubElement(node, me("t"))
    if text.startswith(" ") or text.endswith(" "):
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    t.text = text
    return node


def kids(tag, items):
    node = etree.Element(me(tag))
    seq = items if isinstance(items, (list, tuple)) else [items]
    for c in seq:
        node.append(c)
    return node


def sub(base, subscript):
    node = etree.Element(me("sSub"))
    node.append(kids("e", base if isinstance(base, (list, tuple)) else [base]))
    node.append(kids("sub", subscript if isinstance(subscript, (list, tuple)) else [subscript]))
    return node


def sup(base, superscript):
    node = etree.Element(me("sSup"))
    node.append(kids("e", base if isinstance(base, (list, tuple)) else [base]))
    node.append(kids("sup", superscript if isinstance(superscript, (list, tuple)) else [superscript]))
    return node


def frac(num, den):
    node = etree.Element(me("f"))
    etree.SubElement(node, me("fPr"))
    node.append(kids("num", num if isinstance(num, (list, tuple)) else [num]))
    node.append(kids("den", den if isinstance(den, (list, tuple)) else [den]))
    return node


def dmath(*inner, beg="(", end=")"):
    node = etree.Element(me("d"))
    dpr = etree.SubElement(node, me("dPr"))
    b = etree.SubElement(dpr, me("begChr"))
    b.set(f"{{{M}}}val", beg)
    e = etree.SubElement(dpr, me("endChr"))
    e.set(f"{{{M}}}val", end)
    node.append(kids("e", inner))
    return node


def func(name, *args):
    node = etree.Element(me("func"))
    etree.SubElement(node, me("funcPr"))
    fname = etree.SubElement(node, me("fName"))
    fname.append(kids("e", [r(name, "p")]))
    node.append(kids("e", args))
    return node


def nary(chr_val, subsc, supsc, *inner):
    node = etree.Element(me("nary"))
    pr = etree.SubElement(node, me("naryPr"))
    ch = etree.SubElement(pr, me("chr"))
    ch.set(f"{{{M}}}val", chr_val)
    node.append(kids("sub", subsc if isinstance(subsc, (list, tuple)) else [subsc]))
    node.append(kids("sup", supsc if isinstance(supsc, (list, tuple)) else [supsc]))
    node.append(kids("e", inner))
    return node


def om(children):
    math = etree.Element(me("oMath"))
    for c in children:
        math.append(c)
    para = etree.Element(me("oMathPara"))
    pr = etree.SubElement(para, me("oMathParaPr"))
    jc = etree.SubElement(pr, me("jc"))
    jc.set(f"{{{M}}}val", "center")
    para.append(math)
    return para


def eq_motion():
    return om([
        sub(r("p"), [r("i"), r(",", "p"), r("t"), r("+", "p"), r("1", "p")]),
        r(" = ", "p"),
        sub(r("Π", "p"), r("Ω")),
        dmath(
            sub(r("p"), [r("i"), r(",", "p"), r("t")]),
            r(" + ", "p"),
            r("v", "i"),
            sub(r("Δt", "i"), r("t")),
            r(" ", "p"),
            r("d"),
            dmath(sub(r("a"), [r("i"), r(",", "p"), r("t")])),
        ),
    ])


def eq_link():
    return om([
        sub(r("A"), [r("i"), r(",", "p"), r("j"), r(",", "p"), r("t")]),
        r(" = ", "p"),
        r("1", "p"),
        dmath(
            r("∥", "p"),
            sub(r("p"), [r("i"), r(",", "p"), r("t")]),
            r(" − ", "p"),
            sub(r("p"), [r("j"), r(",", "p"), r("t")]),
            r("∥", "p"),
            r(" ≤ ", "p"),
            sub(r("R"), r("c")),
        ),
        r(" ", "p"),
        sub(r("z"), [r("i"), r(",", "p"), r("t")]),
        sub(r("z"), [r("j"), r(",", "p"), r("t")]),
        sub(r("u"), [r("i"), r(",", "p"), r("t")]),
        sub(r("u"), [r("j"), r(",", "p"), r("t")]),
    ])


def eq_ni():
    return om([
        sub(r("𝒩"), r("i")),
        dmath(r("t")),
        r(" = ", "p"),
        dmath(
            r("j ", "p"),
            r("≠ ", "p"),
            r("i ", "p"),
            r("| ", "p"),
            sub(r("A"), [r("i"), r(",", "p"), r("j"), r(",", "p"), r("t")]),
            r(" = ", "p"),
            r("1", "p"),
            beg="{ ",
            end=" }",
        ),
    ])


def eq_p():
    return om([
        sub(r("P"), [r("i"), r(",", "p"), r("t")]),
        r("(x)", "p"),
        r(" = ", "p"),
        frac(
            r("1", "p"),
            [
                r("1", "p"),
                r(" + ", "p"),
                func("exp", r("−", "p"), sub(r("L"), [r("i"), r(",", "p"), r("t")]), r("(x)", "p")),
            ],
        ),
    ])


def eq_l():
    return om([
        sub(r("L"), [r("i"), r(",", "p"), r("t")]),
        r("(x)", "p"),
        r(" = ", "p"),
        sub(r("L"), [r("i"), r(",", "p"), r("t"), r("−", "p"), r("1", "p")]),
        r("(x)", "p"),
        r(" + ", "p"),
        sub(r("ℓ"), [r("i"), r(",", "p"), r("t")]),
        r("(x)", "p"),
        r(" − ", "p"),
        sub(r("L"), r("0")),
    ])


def eq_e():
    return om([
        sub(r("E"), [r("i"), r(",", "p"), r("t")]),
        r("(x)", "p"),
        r(" = ", "p"),
        func(
            "exp",
            r("−", "p"),
            sub(r("k"), r("q")),
            r("|", "p"),
            sub(r("L"), [r("i"), r(",", "p"), r("t")]),
            r("(x)|", "p"),
        ),
    ])


def eq_rfield():
    return om([
        sub(r("R"), [r("i"), r(",", "p"), r("t")]),
        r("(x)", "p"),
        r(" = ", "p"),
        dmath(r("1", "p"), r(" − ", "p"), r("α", "i")),
        sub(r("R"), [r("i"), r(",", "p"), r("t"), r("−", "p"), r("1", "p")]),
        r("(x)", "p"),
        r(" + ", "p"),
        r("α", "i"),
        r(" + ", "p"),
        r("β", "i"),
        r(" ", "p"),
        r("Δ", "p"),
        sub(r("R"), [r("i"), r(",", "p"), r("t")]),
        r("(x)", "p"),
    ])


def eq_w():
    return om([
        sub(r("w"), [r("i"), r(",", "p"), r("j"), r(",", "p"), r("t")]),
        r(" = ", "p"),
        frac(
            [
                sup(
                    r("e"),
                    [
                        r("−", "p"),
                        sub(r("β"), r("d")),
                        r("∥", "p"),
                        sub(r("p"), r("i")),
                        r(" − ", "p"),
                        sub(r("p"), r("j")),
                        r("∥", "p"),
                    ],
                ),
            ],
            [r("1", "p"), r(" + ", "p"), sub(r("g"), [r("j"), r(",", "p"), r("t")])],
        ),
    ])


def eq_jstar():
    return om([
        sup(r("j"), r("⋆", "p")),
        r(" = ", "p"),
        r("arg", "p"),
        r(" max", "p"),
        sub(r(""), r("j")),
        r(" ", "p"),
        sub(r("w"), [r("i"), r(",", "p"), r("j"), r(",", "p"), r("t")]),
        r(" |", "p"),
        sub(r("L"), [r("j"), r(",", "p"), r("t")]),
        r("(x)|", "p"),
    ])


def eq_att():
    inner = [
        frac(
            [sub(r("Q"), r("i")), sup(sub(r("K"), r("k")), r("⊤", "p"))],
            [r("√", "p"), r("d")],
        ),
        r(" + ", "p"),
        func("log", sub(r("χ"), [r("r"), r(",", "p"), r("i"), r(",", "p"), r("k")])),
        r(" − ", "p"),
        sub(r("κ"), r("r")),
        sub(r("d"), [r("i"), r("k")]),
    ]
    return om([
        sub(r("α"), [r("r"), r(",", "p"), r("i"), r(",", "p"), r("k")]),
        r(" = ", "p"),
        func("softmax", dmath(*inner)),
    ])


def eq_reward():
    return om([
        sub(r("r"), r("t")),
        r(" = ", "p"),
        sub(r("λ"), r("D")),
        sub(r("ΔD"), r("t")),
        r(" + ", "p"),
        sub(r("λ"), r("C")),
        sub(r("ΔC"), r("t")),
        r(" − ", "p"),
        sub(r("λ"), [r("safe", "p")]),
        sub(r("c"), r("t")),
    ])


def eq_dt():
    return om([
        sub(r("D"), r("t")),
        r(" = ", "p"),
        frac(
            [r("|", "p"), r("𝒯", "i"), sub(r(""), r("found")), r("|", "p")],
            [sub(r("N"), r("tgt"))],
        ),
    ])


def eq_ct():
    return om([
        sub(r("C"), r("t")),
        r(" = ", "p"),
        frac([r("1", "p")], [r("|", "p"), r("𝒳", "i"), r("|", "p")]),
        nary("∑", r("x", "i"), r("𝒳", "i"), r("|", "p"), r("2", "p"), sub(r("P"), r("t")), r("(x)", "p"), r(" − ", "p"), r("1", "p"), r("|", "p")),
    ])


def eq_t80():
    return om([
        sub(r("T"), r("80")),
        r(" = ", "p"),
        r("min", "p"),
        dmath(r("t ", "p"), r("| ", "p"), sub(r("D"), r("t")), r(" ≥ ", "p"), r("0.8", "p"), beg="{ ", end=" }"),
    ])


def eq_clip():
    return om([
        sup(r("L"), [r("CLIP", "p")]),
        dmath(r("θ")),
        r(" = ", "p"),
        r("Ê", "p"),
        dmath(
            r("min", "p"),
            dmath(
                sub(r("ρ"), r("t")),
                dmath(r("θ")),
                sub(r("Â"), r("t")),
                r(" , ", "p"),
                r("clip", "p"),
                dmath(
                    sub(r("ρ"), r("t")),
                    dmath(r("θ")),
                    r(" , ", "p"),
                    r("1", "p"),
                    r(" − ", "p"),
                    r("ε", "i"),
                    r(" , ", "p"),
                    r("1", "p"),
                    r(" + ", "p"),
                    r("ε", "i"),
                ),
                sub(r("Â"), r("t")),
            ),
        ),
    ])


def eq_rho():
    return om([
        sub(r("ρ"), r("t")),
        dmath(r("θ")),
        r(" = ", "p"),
        frac(
            [sub(r("π"), r("θ")), dmath(sub(r("a"), r("t")), r(" | ", "p"), sub(r("o"), r("t")))],
            [sub(r("π"), [r("θ", "i"), r("old", "p")]), dmath(sub(r("a"), r("t")), r(" | ", "p"), sub(r("o"), r("t")))],
        ),
    ])


def eq_gae():
    return om([
        sub(r("Â"), r("t")),
        r(" = ", "p"),
        nary(
            "∑",
            r("l", "i"),
            r("H", "i"),
            sup(dmath(r("γλ", "i")), r("l")),
            sub(r("δ"), [r("t"), r("+", "p"), r("l")]),
        ),
    ])


def eq_delta():
    return om([
        sub(r("δ"), r("t")),
        r(" = ", "p"),
        sub(r("r"), r("t")),
        r(" + ", "p"),
        r("γ", "i"),
        r("V", "p"),
        dmath(sub(r("s"), [r("t"), r("+", "p"), r("1", "p")])),
        r(" − ", "p"),
        r("V", "p"),
        dmath(sub(r("s"), r("t"))),
    ])


def eq_front():
    return om([
        r("q", "i"),
        dmath(r("x")),
        r(" = ", "p"),
        r("E", "i"),
        dmath(r("x")),
        r(" + ", "p"),
        r("0.15", "p"),
        r(" ", "p"),
        r("P", "i"),
        dmath(r("x")),
        r(" + ", "p"),
        r("0.65", "p"),
        r(" ", "p"),
        r("R", "i"),
        dmath(r("x")),
    ])


def eq_decpomdp():
    return om([
        r("𝒢", "i"),
        r(" = ", "p"),
        dmath(
            r("𝒩", "i"),
            r(" , ", "p"),
            r("𝒮", "i"),
            r(" , ", "p"),
            dmath(sub(r("𝒜"), r("i"))),
            r(" , ", "p"),
            r("P", "p"),
            r(" , ", "p"),
            r("R", "p"),
            r(" , ", "p"),
            dmath(sub(r("𝒪"), r("i"))),
            r(" , ", "p"),
            r("γ", "i"),
        ),
    ])


EQS = {
    "EQMOTION": eq_motion,
    "EQLINK": eq_link,
    "EQNI": eq_ni,
    "EQP": eq_p,
    "EQL": eq_l,
    "EQE": eq_e,
    "EQRFIELD": eq_rfield,
    "EQW": eq_w,
    "EQJSTAR": eq_jstar,
    "EQATT": eq_att,
    "EQREWARD": eq_reward,
    "EQDT": eq_dt,
    "EQCT": eq_ct,
    "EQT80": eq_t80,
    "EQCLIP": eq_clip,
    "EQRHO": eq_rho,
    "EQGAE": eq_gae,
    "EQDELTA": eq_delta,
    "EQFRONT": eq_front,
    "EQDEC": eq_decpomdp,
}


def replace_placeholder(sld, key, builder):
    found = 0
    for t in sld.iter(ae("t")):
        if t.text and t.text.strip() == key:
            p = t
            while p is not None and p.tag != ae("p"):
                p = p.getparent()
            if p is None:
                continue
            pPr = p.find(ae("pPr"))
            for child in list(p):
                p.remove(child)
            if pPr is not None:
                p.insert(0, pPr)
            p.append(builder())
            found += 1
    return found


def main(path):
    prs = Presentation(path)
    total = 0
    for slide in prs.slides:
        for key, builder in EQS.items():
            total += replace_placeholder(slide._element, key, builder)
    prs.save(path)
    print(f"injected {total} OMML equations into {path}")
    if total < 18:
        print("WARNING: expected at least 18 equation placeholders", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1])
