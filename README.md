# Issues to Resolve

* Random seed library selection?
* Representation of graphs?

# Proposed Interface

## implicit graphs

```
struct Vertex {

  vector<double> coords;
  Vertex* parent // initially self-pointer;
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

Vertex generateRandomVertex(dimensions) // Instance method

Graph generateGraph(int size, int dimensions) {



  for (int i = 0; i < size; i++) {

  }
  // Generate vector<Vertex*> vertices

  // Make singletons

  // Double for loop generates edges
  void calcEuclideanDist(Edge* e)

  for (int i = 0; i < size; i++) {


  }

  Graph new_graph = {vertices.length(), edges.length(), edges, vertices};
  return new_graph;
}


int findSet() // Index in array of sets
Set* makeSet(Vertex vertices[]) // Builds set from list of vertices
Set* linkSet(a,b) // Finds a and b and unions them
```

generateGraphFromSpecVertices
sortVertices


## explicit graphs
