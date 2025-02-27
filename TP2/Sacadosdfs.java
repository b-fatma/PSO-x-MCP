import java.util.*;

class obj {
    int val, pds;

    public obj(int val, int pds) {
        this.val = val;
        this.pds = pds;
    }
}

public class Sacados {
    static final int pds_max = 50;
    static int best_val;
    static boolean[] best_sol;

    public static void main(String[] args) {
        int[] tailles = {8, 10, 12, 15}; // tailles à tester
        
        for (int taille : tailles) {
            System.out.println("\n🔹 test avec " + taille + " obj :");
            test_dfs(taille);
        }
    }

    public static void test_dfs(int nb_obj) {
        obj[] objs = gen_obj(nb_obj);
        aff_obj(objs);

        best_val = 0;
        best_sol = null;
        boolean[] sol = new boolean[nb_obj];

        long start = System.nanoTime();
        dfs(objs, 0, sol, 0, 0);
        long end = System.nanoTime();

        if (best_sol != null) {
            aff_sol(best_sol);
            System.out.println("✅ sol opt trouvée !");
            System.out.println("pds total : " + calc_pds(best_sol, objs));
            System.out.println("val total : " + best_val);
        } else {
            System.out.println("❌ aucune sol valide.");
        }

        double time = (end - start) / 1e6;
        System.out.println("⏱ temps exec : " + time + " ms");
    }

    public static obj[] gen_obj(int nb) {
        Random r = new Random();
        obj[] objs = new obj[nb];
        for (int i = 0; i < nb; i++) {
            objs[i] = new obj(r.nextInt(20) + 1, r.nextInt(20) + 1);
        }
        return objs;
    }

    public static void aff_obj(obj[] objs) {
        System.out.println("obj générés :");
        for (int i = 0; i < objs.length; i++) {
            System.out.println("obj " + (i + 1) + " - val : " + objs[i].val + ", pds : " + objs[i].pds);
        }
    }

    public static void dfs(obj[] objs, int idx, boolean[] sol_act, int pds_act, int val_act) {
        if (pds_act > pds_max) return;

        if (idx == objs.length) {
            if (val_act > best_val) {
                best_val = val_act;
                best_sol = Arrays.copyOf(sol_act, sol_act.length);
            }
            return;
        }

        dfs(objs, idx + 1, sol_act, pds_act, val_act);

        sol_act[idx] = true;
        dfs(objs, idx + 1, sol_act, pds_act + objs[idx].pds, val_act + objs[idx].val);
        sol_act[idx] = false;
    }

    public static void aff_sol(boolean[] sol) {
        System.out.println("\nsol opt trouvée :");
        for (boolean pris : sol) {
            System.out.print(pris ? "1 " : "0 ");
        }
        System.out.println();
    }

    public static int calc_pds(boolean[] sol, obj[] objs) {
        int pds_tot = 0;
        for (int i = 0; i < sol.length; i++) {
            if (sol[i]) {
                pds_tot += objs[i].pds;
            }
        }
        return pds_tot;
    }
}
