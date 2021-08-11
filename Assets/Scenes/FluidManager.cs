using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using UnityEngine;

public class FluidManager : MonoBehaviour
{
    int ParticleNum = 32 * 32 * 32;
    // Start is called before the first frame update
    public Material material;

    int stride;
    int warpCount;
    const int WARP_SIZE = 1024;

    ComputeBuffer particles;


    Vector3[] initBuffer;

    void Start()
    {
        warpCount = Mathf.CeilToInt((float)ParticleNum / WARP_SIZE);

        stride = Marshal.SizeOf(typeof(Vector3));

        particles = new ComputeBuffer(ParticleNum, stride);

        initBuffer = new Vector3[ParticleNum];

        for (int i = 0; i < ParticleNum; i++)
        {
            initBuffer[i] = new Vector3(i * 0.1f, 0.0f, 0.0f);
        }

        particles.SetData(initBuffer);

        material.SetBuffer("Particles", particles);
    }

    void OnRenderObject()
    {
        material.SetPass(0);
        Graphics.DrawProceduralNow(MeshTopology.Points, 1, ParticleNum);
    }
}
