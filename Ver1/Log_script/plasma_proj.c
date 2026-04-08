#include <cvode/cvode_proj.h>
#include <nvector/nvector_serial.h>

static int g_ie;
static sunrealtype g_cef, g_nef;

void proj_set_params(int ie, double ce_floor, double neps_floor)
{
    g_ie  = ie;
    g_cef = (sunrealtype)ce_floor;
    g_nef = (sunrealtype)neps_floor;
}

int proj_nonneg(sunrealtype t, N_Vector ycur, N_Vector corr,
                sunrealtype epsProj, N_Vector err, void *user_data)
{
    sunrealtype *y = N_VGetArrayPointer(ycur);
    sunrealtype *c = N_VGetArrayPointer(corr);

    N_VConst(SUN_RCONST(0.0), corr);

    if (y[0] < g_cef)
        c[0] = g_cef - y[0];

    if (y[g_ie] < g_nef)
        c[g_ie] = g_nef - y[g_ie];

    return 0;
}
