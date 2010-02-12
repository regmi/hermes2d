#include <cmath>
#include "../common.h"
#include "c99_functions.h"

#ifdef IMPLELENT_C99

/* constants */
const unsigned long long _NAN = 0x7fffffffffffffffL; //NAN according to IEEE specification
PUBLIC_API const double NAN = *(double*)&_NAN;

/* functions */
PUBLIC_API double exp2(double x)
{
	return pow(2.0, x);
}

PUBLIC_API double log2(double x)
{
	return log(x) / M_LN2;
}

PUBLIC_API double cbrt(double x)
{
	if (!_isnan(x))
		return pow(x, 1.0 / 3.0);
	else
		return x;
}

#endif
