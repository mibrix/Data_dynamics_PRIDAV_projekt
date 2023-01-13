drop table if exists important;

create table important as
SELECT katedra ,meno_skratka  FROM persons WHERE exists(SELECT 1 FROM (SELECT katedra as k,count(katedra) >= 10 as c FROM persons GROUP BY katedra) WHERE katedra = k AND c = 1);
SELECT * FROM important;





drop table if exists labels;
CREATE TABLE labels(id INTEGER PRIMARY KEY,
                      katedra);
INSERT INTO labels(katedra) SELECT distinct katedra FROM important;
SELECT *  FROM labels;