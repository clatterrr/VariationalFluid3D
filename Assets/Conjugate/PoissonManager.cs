using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PoissonManager : MonoBehaviour
{

    public int Nx = 8;
    public int GridNum;
    int smallGridNum;

    public ComputeBuffer bVec;
    public ComputeBuffer xVec;
    public ComputeBuffer AmatCenter;
    public ComputeBuffer AmatDown; // left bottom near
    public ComputeBuffer AmatUp; // right up far
    public int warpCount;
    int warpCountsmall;


    ComputeBuffer dVec;
    ComputeBuffer rVec;
    ComputeBuffer AxVec;
    ComputeBuffer tempVec;


    ComputeBuffer solid;



    float[] dotResultSmall;
    float[] dotResultLarge;
    ComputeBuffer dotBufferFine;
    ComputeBuffer dotBufferSmall;

    int stride;
    int kernelIndex;
    const int WARP_SIZE = 512;
    float[] zeroData;
    float[] nonZeroData;
    Vector3[] Data3d;

    public ComputeShader InitAmat;
    public ComputeShader ScaleAdd;
    public ComputeShader ScaleDot;
    public ComputeShader Reduction;
    public ComputeShader computeAx;


    private void Start()
    {
        Nx = 16;
        GridNum = Nx * Nx * Nx;
        smallGridNum = GridNum / 8;
        warpCount = Mathf.CeilToInt((float)GridNum / WARP_SIZE) + 1;
        warpCountsmall = Mathf.CeilToInt((float)smallGridNum / 64) + 1;
        dotResultLarge = new float[GridNum];
        dotResultSmall = new float[smallGridNum];

        stride = System.Runtime.InteropServices.Marshal.SizeOf(typeof(float));
        bVec = new ComputeBuffer(GridNum, stride);
        xVec = new ComputeBuffer(GridNum, stride);
        dVec = new ComputeBuffer(GridNum, stride);
        rVec = new ComputeBuffer(GridNum, stride);
        solid = new ComputeBuffer(GridNum, stride);
        AmatCenter = new ComputeBuffer(GridNum, stride);
        AxVec = new ComputeBuffer(GridNum, stride);
        tempVec = new ComputeBuffer(GridNum, stride);
        dotBufferFine = new ComputeBuffer(GridNum, stride);
        dotBufferSmall = new ComputeBuffer(smallGridNum, stride);

        stride = System.Runtime.InteropServices.Marshal.SizeOf(typeof(Vector3));
        AmatUp = new ComputeBuffer(GridNum, stride);
        AmatDown = new ComputeBuffer(GridNum, stride);



        zeroData = new float[GridNum];
        nonZeroData = new float[GridNum];
        Data3d = new Vector3[GridNum];


        for (int i = 0; i < GridNum; i++)
        {
            zeroData[i] = 0;
            nonZeroData[i] = 0;
        }
        bVec.SetData(nonZeroData);
        xVec.SetData(zeroData);
        dVec.SetData(zeroData);
        rVec.SetData(zeroData);
        solid.SetData(zeroData);
        AxVec.SetData(zeroData);

        kernelIndex = InitAmat.FindKernel("CSMain");// y = x + a * Amat
        InitAmat.SetInt("Nx", Nx);
        InitAmat.SetBuffer(kernelIndex, "AmatCenter", AmatCenter);
        InitAmat.SetBuffer(kernelIndex, "AmatUp", AmatUp);
        InitAmat.SetBuffer(kernelIndex, "AmatDown", AmatDown);
        InitAmat.SetBuffer(kernelIndex, "rhs", bVec);
        InitAmat.SetBuffer(kernelIndex, "SolidPhi", solid);
        InitAmat.Dispatch(kernelIndex, warpCount, 1, 1);

        //  AmatDown.GetData(Data3d);
        // Debug.Log("wa" + Data3d[1]);

        Solve();

    }
    public void InitBuffer()
    {
        GridNum = Nx * Nx * Nx;
        smallGridNum = GridNum / 8;
        dotResultLarge = new float[GridNum];
        dotResultSmall = new float[smallGridNum];
        warpCount = Mathf.CeilToInt((float)GridNum / WARP_SIZE);
        warpCountsmall = Mathf.CeilToInt((float)smallGridNum / 64);
        stride = System.Runtime.InteropServices.Marshal.SizeOf(typeof(float));
        xVec = new ComputeBuffer(GridNum, stride);
        dVec = new ComputeBuffer(GridNum, stride);
        rVec = new ComputeBuffer(GridNum, stride);
        AxVec = new ComputeBuffer(GridNum, stride);
        tempVec = new ComputeBuffer(GridNum, stride);
        dotBufferFine = new ComputeBuffer(GridNum, stride);
        dotBufferSmall = new ComputeBuffer(smallGridNum, stride);

        zeroData = new float[GridNum];
        for (int i = 0; i < GridNum; i++) zeroData[i] = 0;
        xVec.SetData(zeroData);

    }
    public void Solve()
    {
        kernelIndex = computeAx.FindKernel("CSMain");// y = x + a * Amat
        computeAx.SetBuffer(kernelIndex, "AmatCenter", AmatCenter);
        computeAx.SetBuffer(kernelIndex, "AmatUp", AmatUp);
        computeAx.SetBuffer(kernelIndex, "AmatDown", AmatDown);
        computeAx.SetBuffer(kernelIndex, "xVec", xVec);
        computeAx.SetBuffer(kernelIndex, "yVec", AxVec);
        computeAx.SetInt("Nx", Nx);
        computeAx.Dispatch(kernelIndex, warpCount, 1, 1);

        // residual = rhs - Ax
        kernelIndex = ScaleAdd.FindKernel("CSMain");// Vec2 = Factor0 * Vec0 + Factor1 * Vec1
        ScaleAdd.SetFloat("FactorZero", -1);
        ScaleAdd.SetFloat("FactorOne", 1);
        ScaleAdd.SetBuffer(kernelIndex, "VecZero", AxVec);
        ScaleAdd.SetBuffer(kernelIndex, "VecOne", bVec);
        ScaleAdd.SetBuffer(kernelIndex, "VecTwo", rVec);
        ScaleAdd.Dispatch(kernelIndex, warpCount, 1, 1);

        AxVec.GetData(dotResultLarge);
        for (int i = 0; i < GridNum; i++)
        {
            // if (dotResultLarge[i] != 0) ;
            // Debug.Log("i = " + i + " = " + dotResultLarge[i]);
        }

        // direction = residual
        kernelIndex = ScaleAdd.FindKernel("CSMain");// y = xFactor * x + yFactor * y
        ScaleAdd.SetFloat("FactorZero", 1);
        ScaleAdd.SetFloat("FactorOne", 0);
        ScaleAdd.SetBuffer(kernelIndex, "VecZero", rVec);
        ScaleAdd.SetBuffer(kernelIndex, "VecOne", rVec);
        ScaleAdd.SetBuffer(kernelIndex, "VecTwo", dVec);
        ScaleAdd.Dispatch(kernelIndex, warpCount, 1, 1);

        float rho = 0;
        kernelIndex = ScaleDot.FindKernel("CSMain");
        ScaleDot.SetFloat("FactorZero", 1.0f);
        ScaleDot.SetBuffer(kernelIndex, "VecZero", rVec);
        ScaleDot.SetBuffer(kernelIndex, "VecOne", rVec);
        ScaleDot.SetBuffer(kernelIndex, "dotBuffer512", dotBufferFine);
        ScaleDot.Dispatch(kernelIndex, warpCount, 1, 1);

        kernelIndex = Reduction.FindKernel("CSMain");
        Reduction.SetInt("Nx", Nx);
        Reduction.SetBuffer(kernelIndex, "dot512", dotBufferFine);
        Reduction.SetBuffer(kernelIndex, "dot64", dotBufferSmall);
        Reduction.Dispatch(kernelIndex, 64, 1, 1);

        dotBufferSmall.GetData(dotResultSmall);
        for (int j = 0; j < smallGridNum; j++) rho += dotResultSmall[j];
        //  bVec.GetData(dotResultLarge);
        // for (int j = 0; j < GridNum; j++) rho += dotResultLarge[j];


        float rho_old = rho, alpha, beta, dAd;
        int ite = 0;
        int ite_max = 100;

        while (ite < ite_max)
        {
            ite += 1;
            if (rho < 1e-10) break;
            // A d
            kernelIndex = computeAx.FindKernel("CSMain");// y = x + a * Amat
            computeAx.SetBuffer(kernelIndex, "xVec", dVec);
            computeAx.SetBuffer(kernelIndex, "yVec", AxVec);
            computeAx.Dispatch(kernelIndex, warpCount, 1, 1);

            // d A d
            kernelIndex = ScaleDot.FindKernel("CSMain");
            ScaleDot.SetFloat("FactorZero", 1.0f);
            ScaleDot.SetBuffer(kernelIndex, "VecZero", dVec);
            ScaleDot.SetBuffer(kernelIndex, "VecOne", AxVec);
            ScaleDot.SetBuffer(kernelIndex, "dotBuffer512", dotBufferFine);
            ScaleDot.Dispatch(kernelIndex, warpCount, 1, 1);



            kernelIndex = Reduction.FindKernel("CSMain");
            Reduction.SetInt("Nx", Nx);
            Reduction.SetBuffer(kernelIndex, "dot512", dotBufferFine);
            Reduction.SetBuffer(kernelIndex, "dot64", dotBufferSmall);
            Reduction.Dispatch(kernelIndex, warpCountsmall, 1, 1);



            dotBufferSmall.GetData(dotResultSmall);
            // 至少比循环512次好
            dAd = 0;
            for (int j = 0; j < smallGridNum; j++) dAd += dotResultSmall[j];
            //    Debug.Log("dAd = " + dAd);


            //Debug.Log("dAd = " + rho);
            alpha = rho / dAd;
            //         Debug.Log("alpha = " + alpha);

            // x = x + alpha * d
            kernelIndex = ScaleAdd.FindKernel("CSMain");
            ScaleAdd.SetFloat("FactorZero", alpha);
            ScaleAdd.SetFloat("FactorOne", 1);
            ScaleAdd.SetBuffer(kernelIndex, "VecZero", dVec);
            ScaleAdd.SetBuffer(kernelIndex, "VecOne", xVec);
            ScaleAdd.SetBuffer(kernelIndex, "VecTwo", tempVec);
            ScaleAdd.Dispatch(kernelIndex, warpCount, 1, 1);


            ScaleAdd.SetFloat("FactorZero", 0);
            ScaleAdd.SetFloat("FactorOne", 1);
            ScaleAdd.SetBuffer(kernelIndex, "VecZero", tempVec);
            ScaleAdd.SetBuffer(kernelIndex, "VecOne", tempVec);
            ScaleAdd.SetBuffer(kernelIndex, "VecTwo", xVec);
            ScaleAdd.Dispatch(kernelIndex, warpCount, 1, 1);

            // r = r - alpha *  Ad
            kernelIndex = ScaleAdd.FindKernel("CSMain");
            ScaleAdd.SetFloat("FactorZero", -alpha);
            ScaleAdd.SetFloat("FactorOne", 1);
            ScaleAdd.SetBuffer(kernelIndex, "VecZero", AxVec);
            ScaleAdd.SetBuffer(kernelIndex, "VecOne", rVec);
            ScaleAdd.SetBuffer(kernelIndex, "VecTwo", tempVec);
            ScaleAdd.Dispatch(kernelIndex, warpCount, 1, 1);

            ScaleAdd.SetFloat("FactorZero", 0);
            ScaleAdd.SetFloat("FactorOne", 1);
            ScaleAdd.SetBuffer(kernelIndex, "VecZero", tempVec);
            ScaleAdd.SetBuffer(kernelIndex, "VecOne", tempVec);
            ScaleAdd.SetBuffer(kernelIndex, "VecTwo", rVec);
            ScaleAdd.Dispatch(kernelIndex, warpCount, 1, 1);

            // compute beta
            kernelIndex = ScaleDot.FindKernel("CSMain");
            ScaleDot.SetFloat("FactorZero", 1.0f);
            ScaleDot.SetBuffer(kernelIndex, "VecZero", rVec);
            ScaleDot.SetBuffer(kernelIndex, "VecOne", rVec);
            ScaleDot.SetBuffer(kernelIndex, "dotBuffer512", dotBufferFine);
            ScaleDot.Dispatch(kernelIndex, warpCount, 1, 1);

            kernelIndex = Reduction.FindKernel("CSMain");
            Reduction.SetInt("Nx", Nx);
            Reduction.SetBuffer(kernelIndex, "dot512", dotBufferFine);
            Reduction.SetBuffer(kernelIndex, "dot64", dotBufferSmall);
            Reduction.Dispatch(kernelIndex, warpCountsmall, 1, 1);

            dotBufferSmall.GetData(dotResultSmall);
            rho = 0;
            for (int j = 0; j < smallGridNum; j++) rho += dotResultSmall[j];
            //   Debug.Log("rho = " + rho);
            if (Mathf.Abs(rho) < 1e-10) break;

            beta = rho / rho_old;
            //      Debug.Log("beta = " + beta);
            rho_old = rho;

            // direction = residual + beta * direction
            kernelIndex = ScaleAdd.FindKernel("CSMain");
            ScaleAdd.SetFloat("FactorZero", 1);
            ScaleAdd.SetFloat("FactorOne", beta);
            ScaleAdd.SetBuffer(kernelIndex, "VecZero", rVec);
            ScaleAdd.SetBuffer(kernelIndex, "VecOne", dVec);
            ScaleAdd.SetBuffer(kernelIndex, "VecTwo", tempVec);
            ScaleAdd.Dispatch(kernelIndex, warpCount, 1, 1);

            ScaleAdd.SetFloat("FactorZero", 0);
            ScaleAdd.SetFloat("FactorOne", 1);
            ScaleAdd.SetBuffer(kernelIndex, "VecZero", tempVec);
            ScaleAdd.SetBuffer(kernelIndex, "VecOne", tempVec);
            ScaleAdd.SetBuffer(kernelIndex, "VecTwo", dVec);
            ScaleAdd.Dispatch(kernelIndex, warpCount, 1, 1);

        }
        Debug.Log("final rho = " + rho + " and ite = " + ite);

        ValidateResult();

        AxVec.Release();
        dVec.Release();
        rVec.Release();
        tempVec.Release();
    }

    private void ValidateResult()
    {
        kernelIndex = computeAx.FindKernel("CSMain");// y = x + a * Amat
        computeAx.SetBuffer(kernelIndex, "AmatCenter", AmatCenter);
        computeAx.SetBuffer(kernelIndex, "AmatUp", AmatUp);
        computeAx.SetBuffer(kernelIndex, "AmatDown", AmatDown);
        computeAx.SetBuffer(kernelIndex, "xVec", xVec);
        computeAx.SetBuffer(kernelIndex, "yVec", AxVec);
        computeAx.SetInt("Nx", Nx);
        computeAx.Dispatch(kernelIndex, warpCount, 1, 1);

        xVec.GetData(dotResultLarge);
        //Debug.Log("preesure = " + dotResultLarge[Nx * Nx * 2 + Nx * Nx / 2 + Nx / 2].ToString("f4"));
        for (int i = 0; i < GridNum; i++)
        {
            if (dotResultLarge[i] == 0) continue;
            int ix = i % Nx;
            int iy = i % (Nx * Nx) / Nx;
            int iz = i / (Nx * Nx);
            //Debug.Log("x = " + ix + " y =  " + iy + " z = " + iz + " so pressure = " + dotResultLarge[i].ToString("f4"));
        }
    }
}
