#define RISKFREE 0.02f
#define VOLATILITY 0.30f

kernel void binomial_options_level0(int numSteps,
                                    global const float4* randArray,
                                    global float4* out)
{
    // load shared mem
    unsigned int tid = get_local_id(0);
    unsigned int bid = get_group_id(0);

    __local float4 callA[254+1]; //numSteps+1 elements
    __local float4 callB[254+1]; //numSteps+1 elements

    float4 inRand = randArray[bid];
    float4 s, x, optionYears, dt, vsdt, rdt, r, rInv, u, d, pu, pd, puByr, pdByr, profit;

    s.x = (1.0f - inRand.x) * 5.0f + inRand.x * 30.f;
    s.y = (1.0f - inRand.y) * 5.0f + inRand.y * 30.f;
    s.z = (1.0f - inRand.z) * 5.0f + inRand.z * 30.f;
    s.w = (1.0f - inRand.w) * 5.0f + inRand.w * 30.f;

    x.x = (1.0f - inRand.x) * 1.0f + inRand.x * 100.f;
    x.y = (1.0f - inRand.y) * 1.0f + inRand.y * 100.f;
    x.z = (1.0f - inRand.z) * 1.0f + inRand.z * 100.f;
    x.w = (1.0f - inRand.w) * 1.0f + inRand.w * 100.f;

    optionYears.x = (1.0f - inRand.x) * 0.25f + inRand.x * 10.f;
    optionYears.y = (1.0f - inRand.y) * 0.25f + inRand.y * 10.f;
    optionYears.z = (1.0f - inRand.z) * 0.25f + inRand.z * 10.f;
    optionYears.w = (1.0f - inRand.w) * 0.25f + inRand.w * 10.f;

    dt.x = optionYears.x * (1.0f / (float)numSteps);
    dt.y = optionYears.y * (1.0f / (float)numSteps);
    dt.z = optionYears.z * (1.0f / (float)numSteps);
    dt.w = optionYears.w * (1.0f / (float)numSteps);

    vsdt.x = VOLATILITY * sqrt(dt.x);
    vsdt.y = VOLATILITY * sqrt(dt.y);
    vsdt.z = VOLATILITY * sqrt(dt.z);
    vsdt.w = VOLATILITY * sqrt(dt.w);

    rdt.x = RISKFREE * dt.x;
    rdt.y = RISKFREE * dt.y;
    rdt.z = RISKFREE * dt.z;
    rdt.w = RISKFREE * dt.w;

    r.x = exp(rdt.x);
    r.y = exp(rdt.y);
    r.z = exp(rdt.z);
    r.w = exp(rdt.w);

    rInv.x = 1.0f / r.x;
    rInv.y = 1.0f / r.y;
    rInv.z = 1.0f / r.z;
    rInv.w = 1.0f / r.w;

    u.x  = exp(vsdt.x);
    u.y  = exp(vsdt.y);
    u.z  = exp(vsdt.z);
    u.w  = exp(vsdt.w);

    d.x = 1.0f / u.x;
    d.y = 1.0f / u.y;
    d.z = 1.0f / u.z;
    d.w = 1.0f / u.w;

    pu.x= (r.x - d.x)/(u.x - d.x);
    pu.y= (r.y - d.y)/(u.y - d.y);
    pu.z= (r.z - d.z)/(u.z - d.z);
    pu.w= (r.w - d.w)/(u.w - d.w);

    pd.x = 1.0f - pu.x;
    pd.y = 1.0f - pu.y;
    pd.z = 1.0f - pu.z;
    pd.w = 1.0f - pu.w;

    puByr.x = pu.x * rInv.x;
    puByr.y = pu.y * rInv.y;
    puByr.z = pu.z * rInv.z;
    puByr.w = pu.w * rInv.w;

    pdByr.x= pd.x * rInv.x;
    pdByr.y= pd.y * rInv.y;
    pdByr.z= pd.z * rInv.z;
    pdByr.w= pd.w * rInv.w;

    profit.x = s.x * exp(vsdt.x * (2.0f * tid - (float)numSteps)) - x.x;
    profit.y = s.y * exp(vsdt.y * (2.0f * tid - (float)numSteps)) - x.y;
    profit.z = s.z * exp(vsdt.z * (2.0f * tid - (float)numSteps)) - x.z;
    profit.w = s.w * exp(vsdt.w * (2.0f * tid - (float)numSteps)) - x.w;

    callA[tid].x = profit.x > 0 ? profit.x : 0.0f;
    callA[tid].y = profit.y > 0 ? profit.y : 0.0f;
    callA[tid].z = profit.z > 0 ? profit.z: 0.0f;
    callA[tid].w = profit.w > 0 ? profit.w: 0.0f;

    barrier(CLK_LOCAL_MEM_FENCE);

    for(int j = numSteps; j > 0; j -= 2)
    {
        if(tid < j)
        {
            callB[tid].x = puByr.x * callA[tid].x + pdByr.x * callA[tid + 1].x;
            callB[tid].y = puByr.y * callA[tid].y + pdByr.y * callA[tid + 1].y;
            callB[tid].z = puByr.z * callA[tid].z + pdByr.z * callA[tid + 1].z;
            callB[tid].w = puByr.w * callA[tid].w + pdByr.w * callA[tid + 1].w;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if(tid < j - 1)
        {
            callA[tid].x = puByr.x * callB[tid].x + pdByr.x * callB[tid + 1].x;
            callA[tid].y = puByr.y * callB[tid].y + pdByr.y * callB[tid + 1].y;
            callA[tid].z = puByr.z * callB[tid].z + pdByr.z * callB[tid + 1].z;
            callA[tid].w = puByr.w * callB[tid].w + pdByr.w * callB[tid + 1].w;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write result for this block to global mem
    if(tid == 0) out[bid] = callA[0];

}
