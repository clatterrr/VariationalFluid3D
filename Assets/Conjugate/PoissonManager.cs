using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PoissonManager : MonoBehaviour
{

    const int Nx = 8;
    int GridNum = Nx * Nx * Nx;

    public ComputeBuffer bVec;
    public ComputeBuffer xVec;
    public ComputeBuffer AmatCenter;
    public ComputeBuffer AmatDown; // left bottom near
    public ComputeBuffer AmatUp; // right up far
    public int warpCount;


    ComputeBuffer dVec;
    ComputeBuffer rVec;
    ComputeBuffer AxVec;
    ComputeBuffer tempVec;


    ComputeBuffer solid;

    

    float[] dotResult = new float[64];
    ComputeBuffer dotBuffer512;
    ComputeBuffer dotBuffer64;

    int stride;
    int kernelIndex;
    const int WARP_SIZE = 128;
    float[] zeroData;
    float[] nonZeroData;

    public ComputeShader InitAmat;
    public ComputeShader ScaleAdd;
    public ComputeShader ScaleDot;
    public ComputeShader Reduction;
    public ComputeShader computeAx;


    private void Start()
    {
        warpCount = Mathf.CeilToInt((float)GridNum / WARP_SIZE);
        stride = System.Runtime.InteropServices.Marshal.SizeOf(typeof(float));
        bVec = new ComputeBuffer(GridNum, stride);
        xVec = new ComputeBuffer(GridNum, stride);
        dVec = new ComputeBuffer(GridNum, stride);
        rVec = new ComputeBuffer(GridNum, stride);
        solid = new ComputeBuffer(GridNum, stride);
        AmatCenter = new ComputeBuffer(GridNum, stride);
        AxVec = new ComputeBuffer(GridNum, stride);
        tempVec = new ComputeBuffer(GridNum, stride);
        dotBuffer512 = new ComputeBuffer(512, stride);
        dotBuffer64 = new ComputeBuffer(64, stride);

        stride = System.Runtime.InteropServices.Marshal.SizeOf(typeof(Vector3));
        AmatUp = new ComputeBuffer(GridNum, stride);
        AmatDown = new ComputeBuffer(GridNum, stride);



        zeroData = new float[GridNum];
        nonZeroData = new float[GridNum];


        for (int i = 0; i < GridNum; i++)
        {
            zeroData[i] = 0;
            nonZeroData[i] = 0;
            if (i == Nx * Nx * Nx / 2 + Nx * Nx / 2 + Nx / 2)
            {
                nonZeroData[i] = 1;
            }
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
        InitAmat.SetBuffer(kernelIndex, "SolidPhi", solid);
        InitAmat.Dispatch(kernelIndex, warpCount, 1, 1);


    }
    public  void InitBuffer()
    {
        warpCount = Mathf.CeilToInt((float)GridNum / WARP_SIZE);
        stride = System.Runtime.InteropServices.Marshal.SizeOf(typeof(float));
        xVec = new ComputeBuffer(GridNum, stride);
        dVec = new ComputeBuffer(GridNum, stride);
        rVec = new ComputeBuffer(GridNum, stride);
        AxVec = new ComputeBuffer(GridNum, stride);
        tempVec = new ComputeBuffer(GridNum, stride);
        dotBuffer512 = new ComputeBuffer(512, stride);
        dotBuffer64 = new ComputeBuffer(64, stride);
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
        ScaleDot.SetBuffer(kernelIndex, "dotBuffer512", dotBuffer512);
        ScaleDot.Dispatch(kernelIndex, warpCount, 1, 1);

        kernelIndex = Reduction.FindKernel("CSMain");
        Reduction.SetInt("Nx", Nx);
        Reduction.SetBuffer(kernelIndex, "dot512", dotBuffer512);
        Reduction.SetBuffer(kernelIndex, "dot64", dotBuffer64);
        Reduction.Dispatch(kernelIndex, 64, 1, 1);

        dotBuffer64.GetData(dotResult);
        for (int j = 0; j < 64; j++) rho += dotResult[j];
        float rho_old = rho, alpha, beta, dAd;
        float[] dotResult512 = new float[512];
        for (int i = 0; i < 2; i++)
        {
            // A d
            kernelIndex = computeAx.FindKernel("CSMain");// y = x + a * Amat
            computeAx.SetBuffer(kernelIndex, "xVec", dVec);
            computeAx.SetBuffer(kernelIndex, "yVec", AxVec);
            computeAx.Dispatch(kernelIndex, warpCount, 1, 1);

            // d A d
            kernelIndex = ScaleDot.FindKernel("CSMain");
            ScaleDot.SetFloat("FactorZero", 1);
            ScaleDot.SetBuffer(kernelIndex, "VecZero", dVec);
            ScaleDot.SetBuffer(kernelIndex, "VecOne", AxVec);
            ScaleDot.SetBuffer(kernelIndex, "dotBuffer512", dotBuffer512);
            ScaleDot.Dispatch(kernelIndex, warpCount, 1, 1);



            kernelIndex = Reduction.FindKernel("CSMain");
            Reduction.SetInt("Nx", Nx);
            Reduction.SetBuffer(kernelIndex, "dot512", dotBuffer512);
            Reduction.SetBuffer(kernelIndex, "dot64", dotBuffer64);
            Reduction.Dispatch(kernelIndex, 64, 1, 1);
            /*
            
            dotBuffer512.GetData(dotResult512);
            for(int j = 0;j < 512;j++)
            {
                if (dotResult512[j] != 0)
                    Debug.Log("j = " + j + " = " + dotResult512[j]);
            }
            */
            dotBuffer64.GetData(dotResult);
            // 至少比循环512次好
            dAd = 0;
            for (int j = 0; j < 64; j++) dAd += dotResult[j];
            Debug.Log("dAd = " + dAd);



            alpha = rho / dAd;
            Debug.Log("alpha = " + alpha);

            // x = x + alpha * d
            kernelIndex = ScaleAdd.FindKernel("CSMain");
            ScaleAdd.SetFloat("FactorZero", alpha);
            ScaleAdd.SetFloat("FactorOne", 1);
            ScaleAdd.SetBuffer(kernelIndex, "VecZero", dVec);
            ScaleAdd.SetBuffer(kernelIndex, "VecOne", xVec);
            ScaleAdd.SetBuffer(kernelIndex, "VecTwo", xVec);
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
            ScaleDot.SetBuffer(kernelIndex, "dotBuffer512", dotBuffer512);
            ScaleDot.Dispatch(kernelIndex, warpCount, 1, 1);

            kernelIndex = Reduction.FindKernel("CSMain");
            Reduction.SetInt("Nx", Nx);
            Reduction.SetBuffer(kernelIndex, "dot512", dotBuffer512);
            Reduction.SetBuffer(kernelIndex, "dot64", dotBuffer64);
            Reduction.Dispatch(kernelIndex, 64, 1, 1);

            dotBuffer64.GetData(dotResult);
            rho = 0;
            for (int j = 0; j < 64; j++) rho += dotResult[j];
            if (Mathf.Abs(rho) < 1e-10) break;
            beta = rho / rho_old;
            Debug.Log("beta = " + beta);
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
    }
}
