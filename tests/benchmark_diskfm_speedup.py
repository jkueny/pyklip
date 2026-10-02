"""benchmark_diskfm_speedup.py

Benchmark DiskFM forward modelling from a saved KL basis across several
pyklip copies (e.g. stock, this PR, and a MagAO-X reference branch).

A realistic HR 4796A debris disk model (no PSF convolution) is forward
modelled through the KL basis by each pyklip copy in a fresh subprocess, so
the different `pyklip` packages never mix. The runtime of each copy is printed
and each forward model is saved next to the KL basis file.

Not collected by pytest (no test_ prefix) since it needs a large private
KL basis file. Usage:

    python tests/benchmark_diskfm_speedup.py [--klbasis FILE] [--worktrees DIR ...]
                                             [--python EXE] [--niter N]
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import astropy.io.fits as fits

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT_DIR = os.path.dirname(REPO_DIR)

DEFAULT_KLBASIS = os.path.expanduser(
    "~/data/HR4796a_lco2023a_magao-x_20230309_10/raws_20230310T054736_s_lyot_stop/"
    "camsci1/lite_psflib/klip_fm_files/camsci1_i_20230309_10_klbasis.h5")
DEFAULT_WORKTREES = [os.path.join(PARENT_DIR, "pyklip_stock"),
                     REPO_DIR,
                     os.path.join(PARENT_DIR, "pyklip_magaox")]
DEFAULT_PYTHON = os.path.join(PARENT_DIR, "pyklip_magaox", ".venv", "bin", "python")

# HR 4796A, MagAO-X camsci1 i' band, 2023-03-09
# (from HR4796_i_camsci1_20230309_10.yaml, "modified" disk model with hg_2g SPF)
DISK_PARAMS = dict(rc=77.4, alpha_in=45.9, alpha_out=15.0, beta=1.0, a_r=0.01,
                   inc=76.6, pa=25.1, dx=-2.0, dy=1.8, Norm=595.57,
                   g1=0.905, g2=-0.145, alpha1=0.365)
DIM = 224
PIXSCALE = 0.012  # arcsec / pix
DISTANCE = 72.248  # pc

JSON_TAG = "BENCH_JSON:"


def make_disk_model(dim=DIM, pixscale=PIXSCALE, distance=DISTANCE, xstep=0.1,
                    chunk=2048, **params):
    """
    Vectorized port of mod_gen_disk_dxdy_2g (Max Millar-Blanchaer, modified by
    J. Mazoyer): 2-component Henyey-Greenstein SPF, two power-law radial
    density peaking at rc, Gaussian vertical profile. The per-pixel adaptive
    quad integration is replaced by a fixed-grid line-of-sight integration.

    Args:
        dim: dimension of the square image in pixels
        pixscale: pixel scale in arcsec
        distance: distance of the star in pc
        xstep: line-of-sight integration step in au
        chunk: number of pixels integrated at once (bounds memory)
        params: disk parameters, see DISK_PARAMS

    Returns:
        a 2d model of shape (dim, dim), normalized to Norm at 90 deg scattering
    """
    p = dict(DISK_PARAMS, **params)
    rc, m, n, beta, a_r = p["rc"], p["alpha_in"], p["alpha_out"], p["beta"], p["a_r"]
    g1, g2, alpha1 = p["g1"], p["g2"], p["alpha1"]
    R1, R2 = rc - 25., rc + 25.

    # +x is the line of sight, +y is right, +z is up (au)
    xsize = dim / 2. * pixscale * distance
    y = np.linspace(-xsize, xsize, num=dim)
    z = np.linspace(-xsize, xsize, num=dim)
    yp, zp = np.meshgrid(y, z)  # image[j, i] <-> (z[j], y[i])

    incl = np.radians(90 - p["inc"])
    ci, si = np.cos(incl), np.sin(incl)
    pa_rad = np.radians(90 - p["pa"])
    cos_pa, sin_pa = np.cos(pa_rad), np.sin(pa_rad)

    # rotate by the PA, then project with the inclination and offset
    yy = (yp * cos_pa - zp * sin_pa).ravel()
    zz = (yp * sin_pa + zp * cos_pa).ravel()
    y2z2 = yy * yy + zz * zz
    zpci = zz * ci
    zpsi_dx = zz * si - p["dx"]
    yy_dy2 = (yy - p["dy"])**2

    k = 1. / (4 * np.pi)
    g1_2, g2_2 = g1 * g1, g2 * g2
    hg_90 = (k * alpha1 * (1. - g1_2) / (1. + g1_2)**1.5 +
             k * (1 - alpha1) * (1. - g2_2) / (1. + g2_2)**1.5)

    xp = np.arange(-R2, R2 + xstep / 2., xstep)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    image = np.zeros(dim * dim)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for start in range(0, dim * dim, chunk):
            sl = slice(start, start + chunk)
            xx = xp[None, :] * ci + zpsi_dx[sl, None]
            d1 = np.sqrt(yy_dy2[sl, None] + xx * xx)
            inside = (d1 >= R1) & (d1 <= R2)

            d2 = xp[None, :]**2 + y2z2[sl, None]
            cos_phi = xp[None, :] / np.sqrt(d2)
            hg = (k * alpha1 * (1. - g1_2) / (1. + g1_2 - 2 * g1 * cos_phi)**1.5 +
                  k * (1 - alpha1) * (1. - g2_2) / (1. + g2_2 - 2 * g2 * cos_phi)**1.5)
            radial = ((d1 / rc)**(-2 * m) + (d1 / rc)**(-2 * n))**(-0.5)

            zzp = zpci[sl, None] - xp[None, :] * si
            hh = a_r * d1**beta
            vertical = np.exp(-0.5 * zzp * zzp / (hh * hh))

            integrand = np.where(inside, hg * radial * vertical / d2, 0.)
            image[sl] = trapezoid(integrand, xp, axis=1)

    return p["Norm"] * (image.reshape(dim, dim) / a_r) / hg_90


def _git_commit(worktree):
    try:
        return subprocess.run(["git", "-C", worktree, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _label(worktree):
    name = os.path.basename(os.path.normpath(worktree))
    return name[len("pyklip_"):] if name.startswith("pyklip_") else name


def _time_calls(obj, method, model, niter):
    times = []
    for _ in range(niter):
        t0 = time.perf_counter()
        obj.update_disk(model)
        fmout = getattr(obj, method)()
        times.append(time.perf_counter() - t0)
    return fmout, times


def run_worker(args):
    """Forward model the disk with the pyklip found on PYTHONPATH (one worktree)."""
    import pyklip
    worktree = os.path.realpath(args.worktree)
    if not os.path.realpath(pyklip.__file__).startswith(worktree + os.sep):
        raise RuntimeError("imported {0}, expected pyklip from {1}".format(
            pyklip.__file__, worktree))
    from pyklip.fmlib.diskfm import DiskFM

    label = _label(worktree)
    commit = _git_commit(worktree)
    model = fits.getdata(args.model).astype(float)

    t0 = time.perf_counter()
    diskobj = DiskFM(None, None, None, model, basis_filename=args.klbasis,
                     load_from_basis=True)
    t_load = time.perf_counter() - t0
    print("[{0}] KL basis loaded in {1:.1f} s".format(label, t_load), flush=True)

    methods = ["fm_parallelized"]
    if hasattr(diskobj, "fm_parallelized_jit"):
        methods.append("fm_parallelized_jit")

    results = []
    for method in methods:
        fmout, times = _time_calls(diskobj, method, model, args.niter)
        tag = label if method == "fm_parallelized" else label + "_jit"
        t_rest = np.mean(times[1:]) if len(times) > 1 else times[0]
        print("[{0}] {1}: first call {2:.2f} s, mean of next {3} calls {4:.2f} s".format(
            label, method, times[0], len(times) - 1, t_rest), flush=True)

        outfile = os.path.join(args.outdir, "{0}_BenchmarkModel_FM_{1}.fits".format(
            args.prefix, tag))
        hdr = fits.Header()
        hdr["WORKTREE"] = (worktree[-68:], "pyklip copy used")
        hdr["COMMIT"] = commit
        hdr["METHOD"] = method
        hdr["NITER"] = args.niter
        hdr["TLOAD"] = (round(t_load, 3), "s, DiskFM KL basis load")
        hdr["TFIRST"] = (round(times[0], 3), "s, first update_disk + FM call")
        hdr["TMEAN"] = (round(float(t_rest), 3), "s, mean of later calls")
        fits.writeto(outfile, fmout[0], hdr, overwrite=True)

        results.append(dict(label=tag, commit=commit, method=method, t_load=t_load,
                            times=times, t_iter=float(t_rest), fmfile=outfile))
    print(JSON_TAG + json.dumps(results), flush=True)


def run_benchmark(args):
    """Make the disk model, then forward model it with each pyklip copy."""
    prefix = os.path.basename(args.klbasis)
    if prefix.endswith("_klbasis.h5"):
        prefix = prefix[:-len("_klbasis.h5")]
    else:
        prefix = os.path.splitext(prefix)[0]
    outdir = os.path.dirname(os.path.abspath(args.klbasis))

    t0 = time.perf_counter()
    model = make_disk_model()
    model_file = os.path.join(outdir, prefix + "_BenchmarkModel.fits")
    fits.writeto(model_file, model, overwrite=True)
    print("Disk model saved to {0} ({1:.1f} s)".format(model_file, time.perf_counter() - t0))

    results = []
    for worktree in args.worktrees:
        worktree = os.path.realpath(worktree)
        print("\n===== {0} ({1}) =====".format(_label(worktree), worktree), flush=True)
        env = dict(os.environ, PYTHONPATH=worktree)
        cmd = [args.python, os.path.abspath(__file__), "--worker",
               "--worktree", worktree, "--klbasis", args.klbasis, "--model", model_file,
               "--outdir", outdir, "--prefix", prefix, "--niter", str(args.niter)]
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            if line.startswith(JSON_TAG):
                results += json.loads(line[len(JSON_TAG):])
            else:
                print(line, end="", flush=True)
        if proc.wait() != 0:
            print("!! {0} failed with exit code {1}".format(worktree, proc.returncode))

    if not results:
        return
    ref = next((r for r in results if r["label"] == "stock"), results[0])
    ref_fm = fits.getdata(ref["fmfile"])
    print("\n===== Summary (update_disk + forward model per iteration) =====")
    print("{0:<14}{1:<10}{2:>10}{3:>12}{4:>12}{5:>10}{6:>14}".format(
        "pyklip", "commit", "load [s]", "first [s]", "iter [s]", "speedup",
        "max|dFM|/max"))
    for r in results:
        rel_diff = np.nanmax(np.abs(fits.getdata(r["fmfile"]) - ref_fm)) / np.nanmax(np.abs(ref_fm))
        print("{0:<14}{1:<10}{2:>10.1f}{3:>12.2f}{4:>12.2f}{5:>9.2f}x{6:>14.2e}".format(
            r["label"], r["commit"], r["t_load"], r["times"][0], r["t_iter"],
            ref["t_iter"] / r["t_iter"], rel_diff))
    print("(speedup and FM difference are relative to '{0}')".format(ref["label"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    parser.add_argument("--klbasis", default=DEFAULT_KLBASIS, help="saved DiskFM KL basis (.h5)")
    parser.add_argument("--worktrees", nargs="+", default=DEFAULT_WORKTREES,
                        help="pyklip copies to benchmark (first 'stock' is the reference)")
    parser.add_argument("--python", default=DEFAULT_PYTHON if os.path.exists(DEFAULT_PYTHON)
                        else sys.executable, help="interpreter used for every pyklip copy")
    parser.add_argument("--niter", type=int, default=3, help="number of timed FM calls")
    # internal: worker mode, one pyklip copy per subprocess
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worktree", help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--outdir", help=argparse.SUPPRESS)
    parser.add_argument("--prefix", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        run_worker(args)
    else:
        run_benchmark(args)


# the guard is required: multiprocessing (spawn on macOS) re-imports this script
if __name__ == "__main__":
    main()
