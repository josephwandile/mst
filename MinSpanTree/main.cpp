

using namespace std;

// TODO Could have a separate 'subset' struct which just keeps track of rank and parent for every edge in the edges vector

struct Vertex {

  vector<double> coords;
  Vertex* parent;
  int rank;
}

struct Edge {

  Vertex* u, v;
  double distance;
}

struct Graph {
  int V;
  int E;
  vector<Edge*> edges;
  vector<Vertex*> vertices;
}

double generateRandomVal() {

  // TODO Random num generator between 0 and 1
  return 0.0
}

Vertex generateRandomVertex(dimensions) {

  Vertex* vertex; // TODO initially self-pointer;
  vector<double> coords;
  coords.reserve(dimensions);

  for (i = 0; i < dimensions; i++) {

    coords.pushback(generateRandomVal());
  }

  vertex.coords = coords;
  vertex.rank = 0;
  vertex.parent = &vertex;

  return vertex;

}

double calcEuclideanDist(Vertex u, Vertex v) {

  // TODO
  return 0.0
}

Graph generateGraph(int size, int dimensions) {

  vector<Vertex*> vertices;
  vector<Edge*> edges;

  vertices.reserve(size);

  // Generate vertices
  for (int i = 0; i < size; i++) {
    vertices.pushback(generateRandomVertex(dimensions));
  }

  // Instantiate all (n choose 2) edges.
  // TODO edges.reserve();

  for (int i = 0; i < size; i++) {

    for (int j = i + 1; j < size; j++) {

      Vertex u = vertices[i];
      Vertex v = vertices[j];
      double distance = calcEuclideanDist(u,v);
      Edge new_edge = {vertices[i], vertices[j], distance};
      edges.pushback(new_edge);
    }
  }

  // TODO Sort edges. Maybe with custom sort algorithm.

  Graph graph = {vertices.length(), edges.length(), edges, vertices};
  return graph;
}

// TODO It makes sense to keep track of parents using indices or pointers but not both.
// findSet(vector<Edge*> edges, index) {
//   if (edges[i].parent != &edges[i]) {
//     edges[i].parent = findSet(edges, parents_index)
//   }
// }
