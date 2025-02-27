import java.util.*;

class Obj { 
    int val, pds;

    public Obj(int val, int pds) { 
        this.val = val;
        this.pds = pds;
    }
}

class Noeud {
    int idx, val_tot, pds_tot;
    boolean[] sol;

    public Noeud(int idx, int val_tot, int pds_tot, boolean[] sol) {
        this.idx = idx;
        this.val_tot = val_tot;
        this.pds_tot = pds_tot;
        this.sol = sol.clone();
    }
}

public class Sacados_bfs {
    static final int pds_max = 50;

    public static void main(String[] args) {
        int[] tailles = {8, 10, 12, 15}; 
        
        for (int taille : tailles) {
            System.out.println("\n test avec " + taille + " obj :");
            test_bfs(taille);
        }
    }

    public static void test_bfs(int nb_obj) {
        Obj[] objs = gen_obj(nb_obj); 
        aff_obj(objs);

        long start = System.nanoTime();
        boolean[] best_sol = bfs(objs);
        long end = System.nanoTime();

        if (best_sol != null) {
            aff_sol(best_sol);
            System.out.println(" solution optimale trouvee");
            System.out.println("pds total : " + calc_pds(best_sol, objs));
            System.out.println("val total : " + eval_sol(best_sol, objs));
        } else {
            System.out.println(" aucune solution valide.");
        }

        double time = (end - start) / 1e6;
        System.out.println(" temps exec : " + time + " ms");
    }

    public static Obj[] gen_obj(int nb) { 
        Random r = new Random();
        Obj[] objs = new Obj[nb];
        for (int i = 0; i < nb; i++) {
            objs[i] = new Obj(r.nextInt(20) + 1, r.nextInt(20) + 1);
        }
        return objs;
    }

    public static void aff_obj(Obj[] objs) { 
        System.out.println("objets generes  :");
        for (int i = 0; i < objs.length; i++) {
            System.out.println("obj " + (i + 1) + " - val : " + objs[i].val + ", pds : " + objs[i].pds);
        }
    }

    public static boolean[] bfs(Obj[] objs) { 
        Queue<Noeud> queue = new LinkedList<>();
        queue.add(new Noeud(0, 0, 0, new boolean[objs.length]));

        boolean[] best_sol = null;
        int best_val = 0;

        while (!queue.isEmpty()) {
            Noeud curr = queue.poll();

            if (curr.idx == objs.length) {
                if (curr.val_tot > best_val) {
                    best_val = curr.val_tot;
                    best_sol = curr.sol;
                }
                continue;
            }

            queue.add(new Noeud(curr.idx + 1, curr.val_tot, curr.pds_tot, curr.sol));

            if (curr.pds_tot + objs[curr.idx].pds <= pds_max) {
                boolean[] new_sol = curr.sol.clone();
                new_sol[curr.idx] = true;
                queue.add(new Noeud(curr.idx + 1, curr.val_tot + objs[curr.idx].val, 
                                    curr.pds_tot + objs[curr.idx].pds, new_sol));
            }
        }

        return best_sol;
    }

    public static void aff_sol(boolean[] sol) {
        System.out.println("\nsolution optimale trouvee :");
        for (boolean pris : sol) {
            System.out.print(pris ? "1 " : "0 ");
        }
        System.out.println();
    }

    public static int eval_sol(boolean[] sol, Obj[] objs) {
        int val_tot = 0;
        for (int i = 0; i < sol.length; i++) {
            if (sol[i]) {
                val_tot += objs[i].val;
            }
        }
        return val_tot;
    }

    public static int calc_pds(boolean[] sol, Obj[] objs) {
        int pds_tot = 0;
        for (int i = 0; i < sol.length; i++) {
            if (sol[i]) {
                pds_tot += objs[i].pds;
            }
        }
        return pds_tot;
    }
}
