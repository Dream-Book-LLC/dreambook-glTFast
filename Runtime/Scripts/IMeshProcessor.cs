// SPDX-FileCopyrightText: 2026 Unity Technologies and the glTFast authors
// SPDX-License-Identifier: Apache-2.0

using UnityEngine;

namespace GLTFast
{
    /// <summary>
    /// Processes decoded Unity meshes before glTFast retains or instantiates them.
    /// </summary>
    public interface IMeshProcessor
    {
        /// <summary>
        /// Process a decoded mesh and return the mesh glTFast should retain.
        /// Returning a different mesh causes glTFast to destroy the decoded input.
        /// </summary>
        /// <param name="mesh">Decoded source mesh.</param>
        /// <returns>The source mesh or a replacement mesh.</returns>
        Mesh ProcessMesh(Mesh mesh);
    }
}
