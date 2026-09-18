#!/usr/bin/env python3
"""Regenerate the accepted analytic MGPA illustration: permission, selectivity, target.

The stored clouds are moment-matched illustrative points, not EEG observations.
Only this three-panel figure is rendered; no model fitting or empirical inputs
are used. Geometry assertions verify fixed Q and the displayed movement costs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np

BLUE, ORANGE = "#0072B2", "#D55E00"
TEAL, PURPLE = "#247F72", "#8662A6"
DARK, GRAY = "#303942", "#A6ADB5"



def arrow(ax, start, end, color=DARK, lw=1., alpha=1., style="-", curve=0.):
    """Draw a directed correction or decomposition arrow."""
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                alpha=alpha, linestyle=style, shrinkA=1.8,
                                shrinkB=1.8, mutation_scale=7,
                                connectionstyle=f"arc3,rad={curve}"), zorder=6)


def clean(ax):
    """Hide axes decorations for the schematic panels."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def sheet(ax, center, width, height, *, preserved=False):
    """Return a parallel-projection map from local 2D coordinates to the page."""
    center = np.asarray(center)
    matrix = np.array([[width / 8.4, width / 17.5],
                       [-height / 16.0, height / 7.3]])
    project = lambda z: np.asarray(z) @ matrix.T + center
    corners = project([[-3.3,-2.6],[3.3,-2.6],[3.3,2.6],[-3.3,2.6]])
    ax.add_patch(Polygon(corners, closed=True,
                         facecolor="#F0ECF6" if preserved else "#EAF4F1",
                         edgecolor="#BDB0CE" if preserved else "#ABCBC3",
                         lw=.7, zorder=0))
    # A slightly stronger front edge is a depth cue, not a coordinate axis.
    ax.plot(*corners[:2].T, color="#A99BBA" if preserved else "#8EB7AC",
            lw=1., zorder=1)
    return project


def dots(ax, clouds, project, *, size=6., alpha=1., zorder=4):
    """Draw the two source clouds with consistent colors and markers."""
    for u in (1, 0):
        points = project(clouds[u])
        if u == 0:
            ax.scatter(*points.T, s=size, color=BLUE, marker="o",
                       edgecolors="white", linewidths=.2, alpha=alpha, zorder=zorder+1)
        else:
            ax.scatter(*points.T, s=size+1, facecolors="none", edgecolors=ORANGE,
                       linewidths=.5, marker="^", alpha=alpha*.8, zorder=zorder)


def line(ax, project, x, color=DARK, lw=.8, alpha=1.):
    """Draw a fixed-score line in the editable coordinate plane."""
    ax.plot(*project([[x,-2.5],[x,2.5]]).T, color=color, lw=lw, alpha=alpha, zorder=2)


def ambient_panel(ax, clouds, mean_x, *, cloud_alpha=.80):
    """Draw full h=q+Bz off both component sheets, and one decomposition.

    Underlying coordinates are (q1,q2,z1,z2). A linear projection preserves
    decomposition parallelograms and parallelism, not lengths/right angles.
    """
    B = np.eye(4)[:,2:]
    P = B@B.T
    q = np.array([1.06,-.46,0.,0.])
    origin = np.array([.47,.315])
    display = np.array([[.008,-.14,.090,.048],
                        [.29,.085,-.028,.059]])
    project = lambda h: np.asarray(h)@display.T+origin
    full = [q+z@B.T for z in clouds]
    assert np.allclose(q@B,0.)
    for points,z in zip(full,clouds):
        assert np.allclose(points@(np.eye(4)-P),q)
        assert np.allclose(points@B,z)

    # Crossing sheets have one common origin; apparent overlap results from
    # the non-injective display map, not shared preserved/editable directions.
    q_corners=np.array([[a,b,0.,0.] for a,b in [(-.34,-1.20),(-.34,1.20),(1.68,1.20),(1.68,-1.20)]])
    z_corners=np.array([[0.,0.,a,b] for a,b in [(-3.3,-2.6),(3.3,-2.6),(3.3,2.6),(-3.3,2.6)]])
    for corners,face,edge,zorder in [(q_corners,"#EDE5F5","#B19BC7",0),
                                     (z_corners,"#E4F1EC","#91B8AA",1)]:
        ax.add_patch(Polygon(project(corners),closed=True,facecolor=face,
                             edgecolor=edge,lw=.85,alpha=.75,zorder=zorder))
    ax.plot(*project(z_corners[:2]).T,color="#80AFA0",lw=1.,zorder=2)
    ax.scatter(*origin,s=9,color=GRAY,zorder=4)
    ax.text(origin[0]-.035,origin[1]-.065,"0",color=GRAY,fontsize=8,ha="right")

    # All blue/orange dots now represent full ambient embeddings, offset by q.
    dots(ax,full,project,size=6.0,alpha=cloud_alpha,zorder=4)
    # A sufficiently positive retained coordinate keeps BOTH highlighted
    # endpoints visibly off the finite Q sheet in this display projection.
    index=int(np.argmin(np.square(clouds[1]-[1.9,1.7]).sum(axis=1)))
    z=clouds[1][index]
    h=full[1][index]
    hz=P@h
    hprime=q+B@np.array([mean_x,z[1]])
    assert np.allclose(h,q+hz)
    assert np.allclose((np.eye(4)-P)@hprime,q)
    assert np.allclose((np.eye(4)-P)@(hprime-h),0.)
    # Dashed parallelogram: q and Bz are projections, not corrected outputs.
    for a,b,color,style in [(np.zeros(4),q,PURPLE,"-"),
                           (np.zeros(4),hz,TEAL,"-"),
                           (q,h,PURPLE,"--"),(hz,h,TEAL,"--")]:
        ax.plot(*project([a,b]).T,color=color,lw=.75,ls=style,alpha=.8,zorder=5)
    ax.scatter(*project(q),s=22,color=PURPLE,edgecolors="white",linewidths=.4,zorder=8)
    ax.scatter(*project(hz),s=24,color=TEAL,edgecolors="white",linewidths=.4,zorder=8)
    ax.scatter(*project(h),s=31,facecolors="white",edgecolors=DARK,linewidths=.9,zorder=9)
    ax.scatter(*project(hprime),s=28,color=DARK,edgecolors="white",linewidths=.4,zorder=9)
    arrow(ax,project(h),project(hprime),DARK,lw=1.6)
    # The original and corrected embeddings share the same projection q.
    ax.plot(*project([q,hprime]).T,color=PURPLE,lw=.7,ls=":",alpha=.65,zorder=5)
    ax.text(* (project(h)+[.035,-.005]),r"$h$",fontsize=10,color=DARK,zorder=12)
    ax.text(* (project(hprime)+[.025,.055]),r"$h'$",fontsize=10,color=DARK,ha="left",zorder=12)
    ax.text(* (project(q)+[-.040,.025]),r"$q$",fontsize=10,color=PURPLE,ha="right",zorder=12)
    ax.text(* (project(hz)+[.030,-.050]),r"$Bz$",fontsize=9.4,color=TEAL,zorder=12)
    ax.text(.22,.885,r"$Q$",fontsize=13,color=PURPLE)
    ax.text(.805,.12,r"$BZ$",fontsize=12,color=TEAL)
    ax.text(.79,.93,r"$H$",fontsize=12,color=DARK)
    return {"ambient_dimension":4,"B":B.tolist(),"P":P.tolist(),
            "q":q.tolist(),"display_matrix":display.tolist(),
            "h":h.tolist(),"h_prime":hprime.tolist(),"Ph":hz.tolist(),
            "note":"Subspaces are orthogonal in R4. Apparent intersections are display-projection artifacts; no correction onto the BZ sheet is performed."}


def render(output):
    """Render the accepted three-panel analytic figure and verify its geometry."""
    OUT = Path(output)
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9.5,
                         "pdf.fonttype":42,"ps.fonttype":42})
    prior=json.loads((Path(__file__).with_name("theory_clouds.json")).read_text())
    clouds=[np.asarray(z) for z in prior["input_points"]]
    b=np.log(4.)
    mu=b+1.2
    mean_x,zero_x=.6,-b/2
    cost=lambda a: 1.64+(a-mu)**2/4
    outputs={name:[np.column_stack([np.full(len(z),x),z[:,1]]) for z in clouds]
             for name,x in (("mean",mean_x),("zero",zero_x))}
    for name,target in (("mean",mu),("zero",0.)):
        old=np.vstack(clouds)
        new=np.vstack(outputs[name])
        assert np.allclose(old[:,1],new[:,1])
        assert np.allclose(b+2*new[:,0],target)
        assert np.isclose(np.square(new-old).sum(1).mean(),cost(target))
    whole_cost=float(np.square(np.vstack(clouds)-[mean_x,0.]).sum(1).mean())
    assert np.isclose(whole_cost,2.64)

    fig=plt.figure(figsize=(7.4,3.15),facecolor="white")
    specs=[(.018,.205,.312,.655),(.364,.205,.274,.655),(.696,.420,.282,.44)]
    axs=[fig.add_axes(rect) for rect in specs]
    titles=["(a) Where can we edit?","(b) How much to change?","(c) Where should we move?"]
    for ax,(x,_,_,_),title in zip(axs,specs,titles):
        clean(ax)
        ax.set(xlim=(0,1),ylim=(0,1))
        fig.text(x,.908,title,fontsize=9.1,fontweight="bold",va="center")

    # (a) Background source clouds are deliberately quieter than the selected
    # observation, its components, and the Q-preserving correction arrow.
    facts_a=ambient_panel(axs[0],clouds,mean_x,cloud_alpha=.30)
    fig.text(.174,.142,r"$H=Q+BZ$",ha="center",fontsize=10)
    fig.text(.174,.065,r"$Q'=Q$",ha="center",fontsize=10,color=PURPLE)

    # (b) Same conditional Z cloud, two alternative full replacements. The
    # branch labels state how much variation is removed, not movement size.
    ax=axs[1]
    pin=sheet(ax,(.17,.51),.34,.285)
    dots(ax,clouds,pin,size=3.8,alpha=.75)
    ax.text(.16,.74,r"$Z=B^\top H$",ha="center",fontsize=8.7)
    ps=sheet(ax,(.74,.76),.42,.245)
    line(ax,ps,mean_x,TEAL,lw=1.)
    dots(ax,outputs["mean"],ps,size=4.3)
    pw=sheet(ax,(.74,.265),.42,.245)
    point=pw([mean_x,0.])
    ax.scatter(*point,s=38,color=BLUE,edgecolors="white",linewidths=.4,zorder=6)
    ax.scatter(*point,s=58,marker="^",facecolors="none",edgecolors=ORANGE,linewidths=.9,zorder=7)
    arrow(ax,(.34,.55),(.53,.75),DARK,lw=1.,curve=-.12)
    arrow(ax,(.34,.46),(.53,.27),DARK,lw=1.,curve=.12)
    ax.text(.73,.95,"Selective",ha="center",fontsize=9.6,color=TEAL)
    ax.text(.75,.592,"Retain variation",ha="center",fontsize=8.5,color=TEAL)
    ax.text(.73,.449,"Whole block",ha="center",fontsize=9.3,color=DARK)
    ax.text(.75,.096,"Collapse to one point",ha="center",fontsize=8.2,color=DARK)
    fig.text(.501,.143,r"$Z\mid Q=q$",ha="center",fontsize=9.8)

    # (c) The same retained variation, placed at different score levels.
    # Device colors remain blue/orange; alternative target colors are used
    # consistently for output lines and cost-curve markers.
    ax=axs[2]
    pt=sheet(ax,(.515,.50),.85,.70)
    dots(ax,clouds,pt,size=4.,alpha=.105,zorder=1)
    for name,x,color in (("zero",zero_x,PURPLE),("mean",mean_x,TEAL)):
        line(ax,pt,x,color,lw=1.1)
        dots(ax,outputs[name],pt,size=6.3,alpha=.9)
    # One original observation, two alternative projections, less visual clutter.
    z=clouds[1][8]
    arrow(ax,pt(z),pt([mean_x,z[1]]),TEAL,lw=.9,alpha=.8)
    arrow(ax,pt(z),pt([zero_x,z[1]]),PURPLE,lw=.8,alpha=.7,style="--")
    xy_mean=pt([mean_x,2.8])
    xy_zero=pt([zero_x,-2.8])
    ax.text(*(xy_mean+[.025,.065]),r"$s=\mu(q)$",ha="center",fontsize=9.4,color=TEAL)
    ax.annotate("",xy=pt([mean_x,2.45]),xytext=xy_mean+[.025,.025],
                arrowprops=dict(arrowstyle="-",color=TEAL,lw=.7),zorder=6)
    ax.text(*(xy_zero+[-.025,-.020]),r"$s=0$",ha="center",va="top",fontsize=9.4,color=PURPLE)
    fig.text(.837,.396,"Same retained information",ha="center",fontsize=8.3,color=DARK)

    # Expected-cost inset is part of (c), not a second technical story.
    # Its minimum is strictly positive. Only the two targets above are shown.
    inset=fig.add_axes([.735,.142,.225,.175])
    clean(inset)
    aa=np.linspace(-.35,5.2,180)
    inset.plot(aa,cost(aa),color=DARK,lw=1.05)
    for a,color,lab in [(0.,PURPLE,"0"),(mu,TEAL,r"$\mu(q)$")]:
        c=cost(a)
        inset.plot([a,a],[1.04,c],ls=":",lw=.7,color=color,alpha=.8)
        inset.scatter(a,c,s=18,color=color,zorder=3)
        inset.text(a,.92,lab,ha="center",va="top",fontsize=8.4,color=color)
        inset.text(a+.15,c+.12,f"{c:.2f}",ha="left",va="bottom",fontsize=8,color=color)
    inset.set(xlim=(-.65,5.65),ylim=(.68,4.45))
    fig.text(.837,.341,"Expected squared movement",ha="center",fontsize=8.1,color=DARK)
    fig.text(.837,.039,r"$\mu(q)=\mathbb{E}[s\mid Q=q]$",ha="center",fontsize=9)

    handles=[Line2D([],[],color=BLUE,marker="o",linestyle="none",markersize=3.5,label="Device A"),
             Line2D([],[],color=ORANGE,marker="^",markerfacecolor="none",linestyle="none",markersize=4,label="Device B")]
    fig.legend(handles=handles,loc="center",bbox_to_anchor=(.501,.974),ncol=2,
               frameon=False,handletextpad=.35,columnspacing=1.3,fontsize=8.4)
    for ext in ("pdf","png"):
        metadata={"CreationDate":None,"ModDate":None} if ext=="pdf" else {}
        fig.savefig(OUT/f"theory_story.{ext}",dpi=280,metadata=metadata)
    plt.close(fig)

    facts={"purpose":"Analytic illustration, not EEG results","builder":Path(__file__).name,
           "ambient_panel":facts_a,
           "cloud_source":"theory_clouds.json",
           "conditioning":"All clouds illustrate one fixed Q=q; (a) full H, (b,c) editable coordinates Z",
           "source_B_probability_given_q":.8,"means":[[-1.,0.],[1.,0.]],"covariance":[[1.,0.],[0.,1.]],
           "score":{"b":float(b),"w":[2.,0.]},"mean_score":float(mu),
           "output_locations":{"mean":mean_x,"zero":float(zero_x)},
           "expected_squared_movement":{"selective_mean":float(cost(mu)),"selective_zero":float(cost(0.)),"whole_block_mean":whole_cost},
           "scope":["Exact full affine steps; no generic guarantee for iterative nonlinear correction.",
                    "Whole block is a theoretical alternative, not a label for LEACE.",
                    "The two targets retain the same information but may affect a frozen head differently."]}
    (OUT/"theory_story.provenance.json").write_text(json.dumps(facts,indent=2)+"\n")
    return facts


def main():
    """Regenerate the theoretical figure without accessing experiment artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "generated")
    args = parser.parse_args()
    render(args.output)
    print("Verified ambient decomposition, Q preservation, retained coordinates and analytic costs.")


if __name__ == "__main__":
    main()
